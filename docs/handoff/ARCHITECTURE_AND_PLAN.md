# TileFoundry Ascend950PR 整体架构与纵向开发计划

本文规定从原生程序到一个逻辑 kernel 调用的实现路线。源码基线为 f341e47529453bdbc8dd81f2c108d36628fb4b8b；交接标签 handoff-20261112 对应 374de776，后续两次提交补充应用二进制接口（Application Binary Interface，ABI）的输出顺序和实际指令属性、变量绑定身份。
当前已有原生 IR、目标注册、指令与分析设施，以及 ABI 和内容身份的局部实现；调度、符号转换、TileLang 发射与运行尚未连成可执行主链。初始调查记录的 24 项 CPU 测试通过只证明当时测试范围，后续增量按独立验证记录判断；原 40 例与设备验收仍未运行。事实见[源码状态调查](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/STATE_MAP.md:4>)、[合同映射调查](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/CONTRACT_GATE_MATRIX.md:13>)；唯一接口规范是[冻结合同](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:4>)。

## 1. 总体架构：同一个程序、同一份计划、一次调用

作者使用带调度声明的高层中间表示（Scheduled HIR）表达算法与策略：Module 包含 Function，LoopRegion 保留循环界限和循环携带值，ExecutionRegion 包住完整向量计算；tf.pipelined 通过 PipelineSpec 声明流水策略。原生节点是程序本体，配置只选择策略。
正常入口保持 checked_program → analyze(...) → schedule.finalize。编译 pass 管理器（PassManager）在既有 PassPoint 上组织规范化、ConvertHIRToTIR 与目标处理；正规 IR 的类型、作用域、布局和内存效果继续由现有分析设施承接。

| 编译段 | 输入 → 真实输出 | 可信依据 |
|---|---|---|
| 声明与来源保存 | Scheduled HIR → typed PipelineSpec、RegionBindingSummary | 真实 LoopRegion、captures、carried params、初值和 yields 与降低后的节点按对象身份对应 |
| 降低与调度分析 | ConvertHIRToTIR → PrimFunction、Sequential/For/If/Let、ExecutionScope/MeshScope → UnitTree 与规范化调度问题（schedule problem） | 每个源计算、访问和状态效果有明确归属，分析阶段完整表达支持域 |
| 模调度（modulo scheduling） | 完整问题 → 共享求解器赋值 → 唯一解析计划（ResolvedPlan） | 策略、依赖、资源和容量同时成立；独立整数复核与原生来源覆盖共同成立 |
| 完成与物理存储 | 同一 ResolvedPlan + effects + Target facts → 符号完成计划、StoragePlan | 完成关系来自真实最后消费者；事件必须进一步落为真实TIR及TL语句，物理字节、布局与版本取自同一计划 |
| 符号转换 | 原入口 + plan/completion/storage → 私有候选 Module 与 source_map | whole-task 克隆保留算式、绑定、原 guard、状态和迭代代次，实际 TIR 按来源替换关系核对 |
| 显式 TileLang | 私有候选 → TL source、真实解析 PrimFunc、实际 lowering 观察 → NativeArtifact | 从实际 PrimFunc 读取支持模板，将后期 refinement 的存储、同步变化纳入对应证明 |
| 安装与调用 | 完整 VERIFIED 候选 → 实际 Module.functions 替换 → 冻结 artifact → bind/invoke | 安装对象与已验证对象相同；ABI 从真实入口派生；运行时只绑定、分配输出并调用一次 |

直接复用 TF 的 parser/printer/evaluator、SSA、type/layout、access/alias、Target 查询、PassManager 和现有 fresh-process/deadline 监督设施。需补齐来源绑定的完整消费、whole-task UnitTree、完整问题、真实共享求解、唯一 ResolvedPlan、有限 completion、版本后 placement、符号转换、TL 模板与结构读取、实际安装和运行接线；这些能力都接入既有编译路径。
七个旧声明/记录 pass 已确定从正常链移除注册，作为首个纵向切片之前的结构调整。其职责由分析主路径承接：normalize_intent 规范化 PipelineSpec、tail/part 声明和真实节点引用；analyze_native_problem 将访问窗、原 guard 与分片效果纳入完整问题；actual verifier 在实际 TIR 上核对谓词覆盖、part 语义及声明兑现。正常入口不再保留空需求的旧记录协议。现有接线位置见[正常入口](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/schedule/finalize.py:12>)。
原始工作包的参考算法已获准只读借鉴并迁入现有设施：[固定 II 约束核](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/implementation/modulo_kernel.py>)进入 [共享 vendor core](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/vendor/tilelang_z3/core.py>)，TF/TL 的 adapter 共用同一 core；[符号模板](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/implementation/symbolic_template.py>)的 p/k/i 构造与 [原生绑定](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/implementation/native_bindings.py>)的 binder 身份克隆、入口替换思路接入现有 plan/apply 层，以 TF Expr/For/If/Let 实现。
[事务参考](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/implementation/compile_transaction.py>)只指导既有 PassManager/主链的分析、求解、完成、放置、认证与有限重规划编排，不安装另一套 driver。参考中的安装顺序统一服从先完成 actual TL conformance、再 install 的冻结合同。
编译侧进程（CPU plan worker）保留私有候选并产生已决计划；设备运行进程（NPU runtime worker）加载同一 TL 源码，承担真实编译、装载与设备调用。复用 WorkerCommand 的独立进程、共享截止时间和结果身份；跨进程传 artifact descriptor，不传活的 tensor、完整 Python IR 或重新求调度的请求。
两个进程的编排允许运行侧先执行不占设备的compile-only并返回实际lowering、二进制及可观察元数据，再由编译侧认证与安装候选；此时尚无公开artifact、bind或invoke。真实资源字段缺失时保留UNKNOWN，不能由JIT对象空表推导资源为零。CPU 进程不加载 torch_npu，运行进程不加载 TF/ORTools 分析链。

## 2. 调度核心：让计划决定实际执行

调度单元（unit）是完整向量函数（whole VF）、DMA、Cube，或有明确内部效果和时序的有限 composite。VF 内 vld/vmul/reduce/vst 仍参与指令、效果与成本分析，外层求解器不将它们拆成独立流水任务。固定小 panel 可保留内部结构；动态 KV 循环不展开成大图。
UnitTree 持有原生节点、region 引用、起止与步长 Expr、执行域、资源、原谓词、captures/carries/yields。变量名和打印路径只服务阅读；同名变量、同形状 allocation 仍是不同定义。
依赖来自 SSA、真实访问（access）、效果（effect）与 alias 分析：写后读 RAW、读后写 WAR、写后写 WAW、配置/累加器/循环携带状态 STATE，以及同一物理槽的版本复用。read/read 本身不产生数据完成边；无法在支持域中描述的访问在问题构造处给出具名分析结果。

| 决策 | 变量与约束 | 在实际程序中的含义 |
|---|---|---|
| 时间 | tᵤ = II·stageᵤ + phaseᵤ，0 ≤ phaseᵤ < II | stage 是迭代位移，phase 是同一模型周期内的发射位置；II 是模型间隔，不是设备纳秒 |
| 数据/状态 | tᵥ + d·II ≥ tᵤ + L | d 取真实值代次距离，L 表达写可见或读/状态完成成本 |
| 版本复用 | t_next_writer + (V+δ)·II ≥ t_last_reader + L_read | V 与 generation 偏差 δ 共同约束槽覆盖，真实硬件完成由事件实现 |
| 周期资源 | 按程序实例、core、pipe 的环形占用区间建约束，包含跨 II 边界重叠 | 不同 AIV 的私有资源分开，同一实际资源不能重叠占用 |
| 容量 | 每存储域内 Σ(V·aligned_slot_bytes) + scratch + reserved ≤ usable_capacity | 真实布局、padding 与 alias 决定需求，后续 placement 精确落实 |
| 发射偏序 | phase 内对 issue_after 与同域源顺序作拓扑线性扩展 | 发射先后与数据读写完成是两种关系 |

stage 和 buffer version 是独立域：stage 决定不同迭代何时发射，V 决定同时存活的数据槽数；同一时序可能需要不同 V，同一 V 也能容纳不同 stage。max_stage 只限制时间域；max_buffer_versions 只给未单独声明 buffer 的 Auto 上界。显式 Fixed(V) 是等式，Auto(min,max) 是求解域，求解后不另估 V 替换答案。
有手工 stage 的 direct-child 列表保留作者绝对 stage，未标 sibling 取 0；物化时另算 origin 与 shift。no-offset 模式所有 stage 为 0。固定 II 只求指定值；自动 II 在有限边界内搜索。
SAT 表示有经复核的赋值；UNSAT 只说明声明的有限域内无解；UNKNOWN 表示尚未判定，不能据此构造可运行计划。已有可行解与最优性分开记录。
所有 region、replan 和编译步骤共享进程单调时钟的绝对截止时间（monotonic absolute deadline，float 秒）；剩余预算只能减少，None 仅表示显式不设截止。重规划消费同一预算，每次调整都回到完整问题及其硬约束。
符号转换令 N=max(0,ceildiv(stop−lower,step))、shift=stage−origin，并构造 p∈[0,N+max_shift)、k=p−shift、i=lower+k·step；每个完整单元只克隆一次，置于 0≤k<N 与原谓词的有效域内。
ResolvedPlan 包含 source/target 身份、多 region 的原生引用及 lower/stop/step Expr、II/origin/max_shift、每单元 stage/phase/start/issue_rank、每 buffer 的 Fixed/Auto 域与 resolved_versions，以及真实 access/state effects 和语义来源映射。completion、storage、TIR 和 TL 消费这一份结果。

## 3. 完成与存储：从逻辑代次到物理槽

符号完成计划（symbolic completion）由有限完成位置（site）和通道（channel）组成；数量取决于单元、边与最大末读者集合，不随迭代数 N 增长。site 关联 before/after/exit、generation Expr、执行域、pipe 和 guard；channel 关联真实 buffer、生产者与消费者及物理机制。
READY 表示该代写入已可见；FREE 表示该代所有必要末读者已完成。前 V 代无需等旧 FREE，随后第 k 代写入单元等待 k−V 代 FREE；每代末读者都发布 FREE，退出以符号尾循环消费 min(V,N) 代残余通知。N=0 不产生该循环的任务或通知，其他未完成异步输出另有退出 drain。
通知的控制域与计算原谓词分别表达，读执行域蕴含写执行域；generation 与 task ordinal 分开，循环携带值的初代和偏移有明确映射。同 pipe FIFO 只在精确执行域和效果合同成立时代替事件。边delay是模型估计，安全性归真实completion；不能以求解出的时距、源代码串行顺序或CPU解释器顺序执行代替DMA完成。
事件命名空间（event namespace）按实际 core、pipe 对、程序实例及物理 bank 分域；READY/FREE 是逻辑原因，不各占一个互不相干的池。跨核 bank 覆盖和 SID 分片必须对应真实目标机制。
物理放置（storage placement）在版本确定后建立 PhysicalLayout、Requirement、StorageKey 和 LiveWindow：GM ABI、每 AIV 的 UB、Cube 的 L1/L0A/L0B/L0C 各有自己的执行域、布局和容量来源。ND/NZ、fractal padding、alignment、version_stride 与转换 scratch 都进入真实字节需求；scratch 按真实 allocation 及存活关系计费一次，每版本独有临时区与可共享一次临时区分别表达。首条实现可让不同 roots 占用独立区间，同 root 的 V 槽照常循环复用；跨 root 复用随后以 completion happens-before 证明。
容量来自当前 Target facts 的 value/source/status，需求不能反推容量。版本、layout、effects 或 completion 改变时重新放置，符号转换与 TL 发射只读取本次正式 offsets。
地址空间与寄存器空间归属不同，但排除不等于放行：StoragePlan只对GM/UB/L1/L0及显式scratch作字节容量；rmem/carrier改由寄存器预算闸门判定。Target必须以value/source/status声明每执行域可用寄存器数量，256B只是单vreg宽度，不得换算数量，缺项UNSOURCED；lowering必须按生命周期报告实际寄存器类型、dtype/lane与数量。任一侧UNSOURCED返回STORAGE_REGISTER_UNSOURCED，占用超预算返回STORAGE_REGISTER_OVER_BUDGET；只有双侧均为已声明事实且占用≤可用才放行，且不进入byte capacity。在此之前rmem维持UNKNOWN_CAPACITY，independent_ub暂无storage正向认证；S3若依赖跨执行域驻留、spill或16-bit打包，需设备/工具链证据。

## 4. TileLang：实现计划，并观察实际 lowering

TileFoundry 的 PrimFunction 与 TileLang/TVM 的 PrimFunc 是不同阶段的真实 IR。发射器将前者映射为后者的受支持表达域；后端的 tile operations（tile ops）继续承担目标指令 lowering，但不重新决定流水策略。

| 六类可组合模板 | 需要保留的语义 |
|---|---|
| safe copy / view | 真实地址、合法 GM 范围、dtype、tail 与执行域 |
| whole VF element / reduction | 算式、绑定、mask、归约与精度，以及内部效果 |
| Cube matmul | panel、累加器、clear 与精度策略 |
| L0C / FIX copy | AIC→AIV 端点、同 dtype 传输与真实完成 |
| UB→L1 pack / copy | NZ 物理 pitch、scratch、SID 覆盖和 pack→copy 完成边 |
| synchronization / control | Set/Wait/barrier、slot/generation、原 guard 与退出 |

实际 tilelang.compile 从 artifact 读取完整编译变换配置（PassConfig）：tl.enable_auto_schedule=False 与 tl.disable_shared_memory_reuse=True。实际 pass_configs 参数是产物身份的一部分；发射器直接生成 Cube/Vector scope 与 SID。
本机pipeline已核实：关闭自动调度（AutoSchedule）时，InsertSync所在整个分支不执行。TF须自己发出计划要求的真实事件，不能期待TL补齐跨任务同步；依据[实际TL pipeline](</home/s00653124/workspace/tilelang/tilelang/ascend/pipeline.py:103>)。dual copy、SIMD merging、ND→NZ、OOB padding、UB allocation 合并、flag→buf 与 ThreadSync 仍可能改变指令、布局、scratch 和同步。
这些变换按有限精化合同（refinement）登记：允许模板内部合法展开、地址重定位和同 namespace 的 flag 双射；计划中的 stage、发射偏序、V、执行域与精度保持不变。首个 profile 不采用未经登记的 flag→mutex 溢出。
read_actual_tl 读取实际 PrimFunc 的 params/buffer_map、operand 与 binder、常数和 op 属性、dtype/shape/layout/offset、For bounds/step、If/guard、slot/generation、core/SID、Set/Wait 端点及顺序。它按模板对应比较 source→TIR 的替换结果与实际 TL；hash 只绑定内容，不能代替对应证明。
还要保存固定 build 的相关 lowering 后实际 IR 和最终资源/CCE 观察，将额外 scratch、arena/alias、内部同步与 manifest 对齐。尤其 UB→L1 采用显式 PACK_NZ VF→V/MTE3 完成边→NZ copy，让内部 packing 可见于同一计划。
顺序固定为：私有 candidate → emit/parse/refinement 核对 → 完整 VERIFIED → install_entry → 公开导出 → bind/invoke。VERIFIED 是真实来源与实际程序对应的结论；安装前完成它，公开 Module.entry_function() 才代表完整可执行结果。失败保留原入口且不发布候选，commit/export 使用这一结果。
现有同步落地仍有实质缺口：AivFlagSet/AivFlagWait/AivDrain仅通过Target hook暴露，尚无完整ISA契约、TL_SURFACE/renderer与completion→apply注入链；TL已有ascend_set_flag/ascend_wait_flag/ascend_pipe_barrier/ascend_threadfence，可接线但须按真实完成语义选择，不能将drain任意换成fence。见[事件面调查](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/OP_SURFACE.md:11>)。
初始局部属性比较尚不能生成完整证明，探针可构造只剩最终未实现项的假工件；见[工件来源调查](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/CHALLENGE_PROBES.md:11>)。实现重点是完整真实产物通路，现有最终拒绝在此之前继续保留。

## 5. 两个算子如何共享这条链

attention_residual 在 token 行上对每个 V 做均方根归一化（RMSNorm），乘 gamma 与 w 后沿 D 归约并除以 √D，形成伪查询打分；RMS 分母为 √(sum(V²)/D+epsilon)。随后沿 N 做 block softmax，再用原 V 加权，最后计算 (1−alpha)·x + alpha·weighted_V。FP32 中间状态、真实 D 的归一化、BF16/FP16 输出与 tail 都体现在原生计算中。
调度问题包含 V load→score/merge、m/l/U 的 distance=1 递推、weights 驻留、x 预取，以及 V 被 score 和 weighted consumers 共同使用的末读关系。后端需要 DMA、完整 VF 的归约/exp/除法/cast 和显式同步；初版可将 score+merge 保持一个 whole VF。AIV-only补面分两批：先补已有TF语义的声明/发射，再注册当前不存在的算子语义，具体前置见S4。
online、blocked_online、resident_n 分别改变 block 宽度、驻留范围和活跃版本字节；literal 与 factored score 都保留 RMS 因子。token_rows、d_chunk、V 和 prefetch 应改变实际 view、loop、placement 或发射顺序，由同一编译链完成。

MLA 计算 scale·(Q_nope K_nopeᵀ + Q_rope K_ropeᵀ)，在 AIV 上做 online softmax，在 AIC 上做 PV，并以 FP32 m/l/U 合并。rope 项在相应 QK 累加中加入一次，最终归一化与 store 仍属于同一次 MIX 逻辑调用；完整 scores/P 不在 GM 物化。
scaleValue≤0 时取 1/√(Dn+Dr)；V 与 K_nope 的值相同是原数值合同，允许 kernel 复用 K 内容，不要求指针相同。分组查询注意力（Grouped-Query Attention，GQA）与 BSND/BNSD 行映射决定真实地址；右下对齐因果掩码（right-aligned causal mask）使用 j ≤ i + Skv − S。QK accumulator、P/alpha、K/P staging 与 U 有不同末消费者和递推边，调度器不能将它们合成一条笼统的“PV 完成”关系。
MLA进入canonical builder之前必须具有正规的aic执行域、MMA_QK/MMA_PV族及MTE1/FIX/dual-copy/NZ传输模板；当前缺失属于TEMPLATE_MISSING，不能写成硬件TARGET_UNSUPPORTED。后端需要六类模板协作：QK→L0C/FIX→AIV softmax/pack→L1→PV→AIV update/store，AIC/AIV 配对属于一次 runtime invocation。P 的 hi/lo 精度路线带来两组真实 MMA 与临时空间，denominator 仍使用 FP32 P。
输出列片独立计算完整 softmax，只计算和写出自己的列区间，grid 随分片后 jobs 改变；延迟缩放（deferred rescale）将旧 epoch 的全部列、hi/lo PV 完成作为 drain/rescale/clear 的前提。两者通过 native effects、同一 solver 和 completion 表达，不形成另一套算子后端。
fp16是本目标应支持的原始输入/输出dtype，不是可排除域：AR合同[四输入BF16或FP16](</home/s00653124/workspace/TileFoundry_950PR_handoff/work_package/specs/05_operator_implementation.md:61>)、[输出cast回输入dtype](</home/s00653124/workspace/TileFoundry_950PR_handoff/work_package/specs/05_operator_implementation.md:75>)；MLA[同dtype、输出dtype与输入一致](</home/s00653124/workspace/TileFoundry_950PR_handoff/work_package/specs/05_operator_implementation.md:174>)，原20例明确含BF16/FP16。fixture proto与case_matrix也列float16，因此fp16不得记TARGET_UNSUPPORTED。
当前拒绝是TF指令面未注册：ops.py只把bf16纳入load/carrier，VECTOR_DTYPES仍f32，_IN_GMEM_LOAD不接受f16；部署TL copy/vcvt表面支持f32↔f16，ABI也有f16。最小指令形状与bf16一致：f16 GM/UB载入→f16寄存器/transport→f32计算与FP32状态→f32→f16转换→f16 UB/GM存储，不把f16塞进当前f32算术域。无设备证据前不声称native f16 vector compute；后续若硬件证明可直接计算，只作优化路径。reader须按位宽规则化dtype/lane/extent/access范围并核对pset谓词宽度；16-bit x128是否符合本机寄存器打包仍需硬件证据。ABI已dtype-ready。实测见[dtype贯通调查](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/DTYPE_GATE.md:9>)与[向量寄存器模型](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/codegen/vector_model.py:36>)。
比较器阈值不在工作包内，但本机指定比较器自带cann-bench per-dtype表：[入口:7](</home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts/npu_compare.py:7>)按accuracy_mode分派，合同用cann-bench；[阈值表:82](</home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts/cann_bench_compare.py:82>)含float16 2^-10、bfloat16 2^-7、float32/float64 2^-13、hifloat32 2^-11及两档fp8。阈值按actual.dtype选取并把golden cast到同一dtype，因此f32 kernel会被按float32容差判定，原dtype不能由f32结果冒充。MLA p_hi_lo先查表中既有hifloat32键，不自造容差；proto缺precision_thresholds只说明该extra不覆盖浮点主阈值，不等于阈值不可得。
调用约定（ABI）保持原公开输入、scalars 和输出合同；host 仅校验元数据、选择 specialization、分配输出并调用一次，数学与布局转换全部在 kernel 内。原用例与输入身份依据[原始数值输入调查](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/INPUT_FACTS.md:11>)。

## 6. 纵向开发计划：每一步都有可执行的真实产物

每个阶段扩展同一条正常入口，前一阶段能力持续运行。先形成最小闭环，再扩大表达域；不等所有调度与后端模块分别完成才集成。下列为未来验收要求，当前均未完成。
进入 S0 前先完成结构调整：移除七个旧 pass 的正常注册，将声明、tail/part 与真实谓词覆盖职责接到上述规范化、分析及 actual verifier；经正常入口观察到新的 pass 顺序与保留的数值语义后，才开始首个切片。这不是后续清理项。
阶段依赖明确为：S0来源证明与真实计划产出→事件落地及认证→S1符号域→S2软流水→S3动态AIV；AIV路径可继续S4，不等待MIX。S3的MIX往返分支必须先具备aic执行域/MMA/传输，再进入S5。两算子各自原20之前均有独立dtype门禁；阶段S6/S7消费两条已验收路径。
MLA 分为 canonical 数值基线与列片/deferred 两个可独立验收的增量：先固定数学与 MIX 数据通路，再改变并行度和状态时序，便于判断每次性能变化。

### 方法论：条件门与正向可达性

**无条件拒绝 ≠ fail-closed。** 合格的门逐条核对证据：证据齐则产出真实结果并将proof内容绑定进身份；证据缺则指明具体region/访问/依赖边及缺失事实再拒绝。类型、hash和成功状态都不替代逐项证明。
每阶段都要用一个受限合法输入证明正向结果可达，并用对应突变证明错误输入不可达。永久拒绝可以记录实现缺口，但其唯一效果若是阻断全部输入，就是设计缺陷，不能算正确阶段门；不得用未来通用支持域尚缺来拒绝当前已证明的受限域。
当前[source audit后的无条件拒绝](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/resolved.py:343>)使ResolvedPlan永不产出，completion/storage/apply/TL与事件观察随之全部不可达。修正应是依来源审计和赋值复核结果作条件判定，保留缺证据的精确拒绝，将实际proof与source/problem/model身份并入计划内容身份；不能只删除拒绝而省略证明。

### S0：单个静态算术 kernel 的完整编译切片

- **入口：** 真实Scheduled HIR的静态f32 kernel经checked_program、analyze、finalize；先采用已证无跨region/carry且UB区间独立的程序，例如返回(2*x,x)的双输出构造；非流水也经过完整问题、共享solver、单一计划与同一转换服务。
- **产物：** 原生来源映射、stage0/V1完整ResolvedPlan、真实effects/completion/storage、注入FlagSet/FlagWait/Drain的私有候选、实际TL及解析PrimFunc/refinement证据；完整离线compile产出的CCE ELF kernel.aibin、host executable.so、Ascend C、params.json与逐文件manifest；认证后的安装入口和CPU输出。
- **通过条件：** 两组不同输入在实际TIR上得到非平凡正确结果，认证安装后入口与打印/发射对象一致；operand/ABI/配置及事件对应；完整离线compile成功且二进制静态元数据与ABI/target匹配，改operand或删/错配关键同步均使认证失效。真实ResolvedPlan的正向返回、类型/作用域及下述两组验收必须实际执行。
- **刻意不做：** 动态界限、多stage、MIX、原算子、设备绑定/执行与计时。CPU evaluator是验证工具；缺外部CANN/bisheng使离线compile不可用时记录BLOCKED_ENV，保留已完成前缀证据，S0整体不通过；源码生成不降格替代二进制交付。

S0内部顺序是“independent_ub来源审计与静态RAW/物理区间证明 → **产出真实ResolvedPlan** → **事件可发射** → emit/parse对应 → 完整离线compile与静态检视 → 实际refinement认证 → 安装与CPU数值”。事件步骤必须先于该程序的VERIFIED，不能推迟到完整算子。

| S0计划与事件里程碑（顺序执行） | 必须真实存在的产物与机器判据 |
|---|---|
| independent_ub产出ResolvedPlan | 正常作者入口经完整source audit、问题构造、SAT赋值与独立postcheck后，resolve_schedule实际返回stage0/V1的ResolvedPlan对象；真实problem_ref、来源proof、时序/版本和内容身份可读。合法例到达返回点；删/反转RAW或改root/版本的负例在具体缺证据处拒绝。此里程碑先于事件注入，为completion提供可挂载的计划，尚不授予安装/发布资格 |
| 契约与发射面 | FlagSet/FlagWait/Drain具备真实ISA/效果/执行域合同，经既有Target机制进入completion、apply、emitter、reader；候选TIR出现实际语句，TL出现对应set/wait/barrier或经证实的drain序列 |
| LOAD→VF、VF→STORE与退出 | 从实际source RAW建立mte2写→VF读、VF写→mte3读的完成协议，异步输出在退出完成；reader核对真实pipe/namespace/ID/guard，不能仅有completion metadata |
| 配对与负例 | 独立token模型确认每代Set与消费式Wait配对一次；删除wait/set/drain或改端点时native/TL认证失败；不得增加自动调度、host synchronize或模型delay来补缺失事件 |

[PIPE_FACTS](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/completion/model.py:176>)声明mte2/mte3为async、visible_at_issue=False且same_pipe_order=False；即使V1、stage0、串行源程序也不免除上述步骤。CPU算术正确只证明数值前缀，不单独证明设备同步安全。

S0的二进制判据依据[真实离线编译探针](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/AIBIN_PROBE.md:5>)：本机source环境脚本使bisheng进入PATH后2–3秒得到aibin与host so；探针以meta_path拦截确认未import torch_npu，/proc/self/maps仅见wheel自带libstub_ascendcl.so、无真实AscendCL驱动库，全程不占设备；探针实例为3128B的ELF64/CCE Machine 0x1029静态aibin、20152B host so、1424B Ascend C及manifest/params；这些尺寸与hash只标识该次构建，不作为所有kernel的常量阈值。未发现bisheng时失败位于device compile，前端可能已成功。这确立可实现的交付门槛，尚不证明当前TF发射程序已通过该门。

| S0观察层与核对职责 | 必需事实与证据上限 |
|---|---|
| 原生候选：verify_actual_program | 从真实CompletionPlan与候选TIR核对Set/Wait/Drain、pipe/ID/generation/guard及单消费配对；缺失不能靠模型delay补齐 |
| TL及refinement：read_actual_tl | 读取实际解析PrimFunc事件intrinsic/operand/控制域与计划比较，再读取实际pass快照及Ascend C中的事件调用，或登记且有事实依据的等效lowering；以结构和端点对应为准，不以全文关键字计数为准 |
| 完整compile与静态检视 | 后端必须新增显式compile站点，固定消费artifact原PrimFunc、完整双键PassConfig、target/arch与编译参数，返回aibin/so/C源/manifest。内部inspect_compiled_artifact辅助例程读取实际ELF类型、kernel symbol、参数元数据/arg-size与kernel-type，回填同一artifact/refinement manifest，由read_actual_tl合并认证，不另建公开编译链 |
| 二进制同步section | .CrossCoreSync_*是S0的**非必需旁证，仅记录**存在/缺失与原字节；它与计划事件、物理flag槽数的对应目前UNKNOWN，不作事件存在、配对或资源计数证明，缺此section也不单独导致S0失败 |

以下为待实现的稳定结果码与可执行判定：verify_actual_program对计划要求的每个site，从候选TIR找同一端点/代次/控制域的实际同步，缺/多/错返回NATIVE_COMPLETION_MISMATCH；read_actual_tl对同一计划的actual PrimFunc逐项读取，错配返回TL_EVENT_CORRESPONDENCE_MISMATCH。它再消费真实lowering快照/Ascend C，支持的事件模板或已登记等效展开逐项对应才记OBSERVED；无法读取或事件消失且无等效证据返回TL_EVENT_LOWERING_UNOBSERVED，整体conformance不为完整VERIFIED。这些函数不能以emitter自报“已发”替代实际读取。
离线compile站点仅当编译器可发现、完整双键被实际调用、aibin/so/params/manifest存在且hash正确时成功；编译环境不可发现返回COMPILE_ENV_UNAVAILABLE，真实编译失败返回COMPILE_FAILED。inspect_compiled_artifact用readelf/strings读取支持的ABI与kernel-type字段，矛盾返回COMPILED_ABI_MISMATCH，必需字段无法解释返回COMPILED_ABI_UNOBSERVED；结果交read_actual_tl合并。二进制.CrossCoreSync_*不参与上述事件成功条件，不触发事件失败码。
**最小事件通过条件：** 候选TIR→实际PrimFunc的显式同步结构对应、受影响lowering→Ascend C的事件语义可解释、真实双键compile成功。若Ascend C不见事件且没有登记的等效展开证据，事件lowering保持未观察，S0不得完整VERIFIED；有.CrossCoreSync_*也不能补过。二进制ABI/kernel-type检查独立必需，最终硬件同步行为在S3设备窗口验证。
ABI静态检视按该固定build的参数布局规则比较params/out_idx、输出角色、__CCE_KernelArgSize、.ParamSummary_/.ParamInfo_及kernel-type；不能把参数字节数简单当tensor数据字节数相加。readelf/strings只提供实际字段与原始字节，字段含义需固定规则；无法解释的必需ABI事实保持未完成。
artifact内容身份分别纳入TL source、Ascend C、aibin、executable.so与manifest的真实hash/size及关联路径，绑定source/plan/ABI/双键配置、target+arch/target_host、TL/TVM build、实际bisheng/ld.lld/CANN与host ABI/库身份。编译时间戳可使重新编译二进制hash不同，须分开候选语义身份与构建实例身份；cache hit校验同一构建文件，不要求跨冷编译位级确定性，也不以时间戳差异计新候选。

### S1：符号循环，先将动态性落到实际 IR

- **入口：** 同一 HIR kernel 加 LoopRegion，界限保留为 Expr；仍为串行时序和单版本。
- **产物：** 实际 For/If/Let、非零 lower、正整常量 step、tail、captures/carries 的来源对应；UnitTree、问题和 TL 循环均为有限结构。
- **通过条件：** N=0/1/17/1_000_003 时实际 IR 结构计数除常数字面量不变；两个同名 binder、不同 root、原 guard 与 effectful Let 经 CPU 执行保持语义；大 N 用结构统计，小 N 用执行结果验证。
- **刻意不做：** 多版本重叠、动态 step、一般 while 与任意非仿射关系推导。

### S2：软流水，让 stage、II、V 真正改变执行

- **入口：** 同一动态 kernel 使用 tf.pipelined/PipelineSpec、TaskIntent 与 BufferSchedule，经正常分析求解。
- **产物：** 固定/自动 II、manual/auto stage 与 Fixed/Auto V 的真实赋值；实际 p/k/i 循环、slot/generation、有限 READY/FREE 和版本后 placement。
- **通过条件：** stage/phase/start 与发射偏序满足整数约束；V1/V2/V3 的 allocation/slot/TL 一致；N=0/1/V−1/V/V+1/17 及 stage gap>N 时 CPU 输出、token 平衡和尾部完成成立；全流程只消费同一绝对截止时间。
- **刻意不做：** 用模型 II 声称设备收益、跨 root 存储复用优化、任意同步协议搜索。

### S3：动态 AIV 设备执行与完整 MIX 往返

- **入口：** 已认证动态kernel经CPU plan worker→同一descriptor→NPU runtime worker；AIV数值先闭合，MIX另沿同一入口推进Cube→Vector→Cube，前置是aic正规执行/存储可达性及MMA、MTE1/FIX、dual copy、NZ pack/UB→L1族注册完整。
- **产物：** 实际 TL compile 参数、前后期 IR/CCE 与资源记录、已绑定 callable、一次调用输出、MIX SID/布局/完成关系和 launch 观察。
- **通过条件：** 动态AIV覆盖S2边界并输出正确GM数值，其完成允许AIV-only算子继续；MIX分支另证明aic/向量执行域、FP32累加、往返布局/同步和真实资源，再允许MLA进入。两者分别有一次logical invocation、无额外host数学、worker隔离及artifact身份结果。
- **刻意不做：** 只凭一个 Cube 指令宣称 MIX 完成、完整算子性能搜索、多 kernel workspace 调度。

### S4：attention_residual 的完整原 20 例

前置为S0事件链与S3动态AIV，AIV-only缺项按两批交付，均落到同一native builder与发射器：

| 批次 | 能力范围与通过判据 |
|---|---|
| 已有语义的声明/发射 | 给已注册的vcvt/vmul/vmula/vadds/vmuls/vdiv/vsqrt/vdup/vdupv/vcadd补真实TL映射和reader对应，给pset16补ISA契约，支持真实常数含−inf及pad值声明；逐项正常入口产物可读。vcadd只保留其已证语义，不冒充完整沿D归约 |
| 新TF注册/模板与dtype一致 | max/min与exp注册真实operand/effect、类型、CPU语义及TL对应；沿D/N的sum/max归约登记whole-VF composite并声明内部顺序/布局；通用舍入cast和tail初始化补足。G1对f16为三项形状：GM/UB载入与transport、f16↔f32双向转换、f16输出存储，计算走FP32状态；bf16同类carrier/convert。reader按位宽规则化dtype/lane/extent、谓词粒度与转换语义，消除六类expected错配；x128寄存器打包与native f16 compute仍需硬件证据。TL已有调用面不代表TF已具备这些语义 |

**进入原20前的G1–G4门禁：** G1指令面按上述三项形状覆盖f16输入/转换/输出，bf16同理；G2发射与实际reader在真实bf16/fp16程序上逐项一致；G3比较器路径/sha256/版本、accuracy_mode=cann-bench及所用per-dtype阈值条目已登记，禁止修改比较器或阈值、禁止向proto注入precision_thresholds；G4确认该例不是p_hi_lo、carrier打包或online精度语义等不可f32代验类。四者全真才可启动该例原比较器；任一缺失记BLOCKED_DEPENDENCY，禁止自造allclose阈值。
AR的17个BF16例可在能力闭合后先做结构与数值验收；3个FP16例在上述G1三项及所需16-bit寄存器/转换硬件证据齐备前保持BLOCKED_DEPENDENCY，但属待实现声明/注册项，不是TARGET_UNSUPPORTED。f32只能证明控制流、版本、同步和计划兑现，不能替代原例数值通过。

- **入口：** 先有真实BF16与FP16最小作者程序（作者dtype非f32，不改emit字面量）分别跑通完整降低与parse；完整native builder返回Scheduled HIR，沿上述正常链进入原test.main(operator_api)。
- **产物：** 最小BF16降低+parse证据、RMS/score/softmax/weighted sum/门控的真实IR与TL、完整输入ABI、原20个key/seed逐例比较结果、资源与一次调用证据。
- **通过条件：** BF16最小作者程序已跑通、两批AIV能力与dtype门禁通过，原比较器和阈值下20例全通过（含FP16的3例），含非对齐D与两dtype；RMS 定向输入、alpha/epsilon 与 gamma/w 改变影响正确输出；plan 对应、容量、数值与 launch 分别有证据。
- **刻意不做：** canonical 基线尚未闭合就展开全体性能族，或以 transport template 的乘法结果代替算子。

### S5：MLA canonical、列片与 deferred 的连续增量

**入口前置：** S3完整MIX往返及其aic执行域/MMA族、MTE1/FIX/dual copy/NZ transfer已通过；当前实现缺模板就记录TEMPLATE_MISSING，不得据此裁定TARGET_UNSUPPORTED。**原20门禁：** G1–G4在Q/K/V的f16/bf16加载、Cube低精度操作数、FP32 QK/softmax/U累加、P cast/pack、p_hi_lo差分与最终f16/bf16输出转换的TF→TL→reader链上全真；G3比较器路径/sha256/版本、accuracy_mode及所用阈值条目已登记；p_hi_lo优先使用表中hifloat32键，不自造容差，也不得修改比较器或注入proto阈值。5个BF16例需G1–G4全部闭合后进入数值验收；15个FP16例在Cube面、f16三项指令形状、p_hi_lo硬件证据与G3阈值登记齐备前必须记BLOCKED_DEPENDENCY，但不记TARGET_UNSUPPORTED，f32、BF16或结构代验均不得冒充其原例通过。

- **入口：** native MIX builder 经同一编译与调用入口，先执行 canonical，再分别启用列片、hi/lo 与 deferred 策略。
- **产物：** canonical 原 20 例及 GQA/显式 scale/真实 tail 补充结果；列片后的实际 grid/列覆盖；deferred 的 epoch 状态、跨列完成与真实设备对照。
- **通过条件：** aic/MMA及两dtype门禁先通过，canonical保持全部原20数值与单调用；每项增量各自通过原比较器，Dn448 末列片、右下 causal 与旧 epoch 全 PV 完成有实际 IR/运行证据；优化参数改变真实产物。
- **刻意不做：** 以补充诊断替代原 20，或把 canonical 通过记为 deferred 已实现。

### S6：用真实设备结果展开算法族与搜索

- **入口：** 候选策略进入同一native builder→compile→原数值→launch→profile，所有候选共享正式编译语义。每个参与排名的原case须先满足G1–G4。G3登记物为docs/handoff/COMPARATOR_THRESHOLD_REGISTRY.json，validator在首次设备数值前写入，implementer只读；字段={registered_at_utc,operator,case_id,dtype,comparator_module_path,comparator_module_sha256,comparator_version,accuracy_mode,threshold_table_ref,threshold_entry:{threshold,small_value_threshold,small_value_error,cancel_boundary,cancel_zero_threshold},registration_basis,pinned_by}。禁止改比较器或注入proto阈值；缺文件/字段或实际sha变化保持BLOCKED_DEPENDENCY。未闭合dtype或G3登记不得用f32结构结果或自造容差充当数值正确。
- **产物：** attention_residual 三种主驻留族与流水候选、MLA canonical/列片/precision/deferred 候选；actual plan/placement/精度身份；原始设备计时样本、代表矩阵与 winner 邻域。
- **通过条件：** 每个可跑case的候选经原比较器数值正确后才计入排名；每算子至少8个代表各8个实质不同候选、每原case至少3个竞争候选，或对BLOCKED_DEPENDENCY给出准确缺失事实；真实产物去重；编译/分配/oracle在计时外；噪声附近重测，预算与未探索范围明确。
- **刻意不做：** 模拟计时排名、只改名字的候选、以预算耗尽代替算法族完成，或宣称全局设备最优。

### S7：冻结 dispatch、cold/hit 与独立复现

- **入口：** 最终公共operator API从冻结specialization dispatch加载已认证artifact，原runner执行完整40例。同一COMPARATOR_THRESHOLD_REGISTRY.json由validator维护、lead冻结、implementer只读；每执行case须有dtype→threshold_entry及comparator sha/version/accuracy_mode。运行前复核当前文件sha与登记值一致，运行后把registry sha写入最终验收记录；缺失或不匹配时40例保持SPECIFIED_NOT_EXECUTED/BLOCKED_DEPENDENCY，不能以f32或结构代验记录为最终PASS。
- **产物：** 同源冷编译与缓存命中身份、逐 case baseline/winner/竞争候选性能表、源码/工具链/ABI/PassConfig/产物关联，以及独立重建记录。
- **通过条件：** G3登记物存在、字段完整且sha复核通过后，最终原40数值、容量、对应与单调用全部成立；cold/hit对应同一artifact；独立者按交付命令重建并复现，所有正常入口测试实际执行且数量非零。
- **刻意不做：** 运行时从调度 JSON 重规划、临时补建未冻结 specialization、沿用历史日志作为当前源的通过证明。

## 7. 工具链前置与明确不做的范围

部署证据中的 TileLang 0e687c17 与规格参考 558d53b 是不同 build。开发前先对当前安装源码与离线编译做能力探针（capability probe），需要设备确认的部分在阶段 S3 单独完成；不先假定两版本一致，也不盲目更换驱动。

| 需得到的事实 | 探针必须产出的真实证据 | 进入 Target facts 的方式 |
|---|---|---|
| Python/Torch/TL/CANN 与原比较器 | 实际路径、源码/build 身份、导入能力和 npu_compare 来源；原文件及 40 keys 身份 | 当前值、来源与已观察范围；包外比较器位置当前 UNKNOWN |
| 双键PassConfig、离线编译与dtype | 实际compile双键参数/合法键校验、关闭AutoSchedule时InsertSync不运行的路由；aibin/so静态ABI；显式事件可读；BF16/FP16全链能力 | S0必须有事件结构对应与真实编译产物，S3再核设备效果；两dtype全链支持是各原20验收前置 |
| flags / event namespace | 同核 core/pipe、跨核 bank/SID、有效 ID 域、重编号或 flag→buf 实际 IR | 参考的同核 8/跨核 16 仅作待核值；实际机制与池以当前 build 为准 |
| UB→L1 ND2NZ | 当前 pass/codegen 的 physical pitch、scratch 元素数/字节与内部 V→MTE3 完成边 | 参考公式 (rows+1)·cols 按实际 dtype/布局复核；先进入模板 requirements 再 placement |
| dual copy、UB arena 与 ThreadSync | 同 dtype 限制、每 SID view/guard 覆盖、实际 arena/alias 与内部线程同步作用域 | 关联真实执行域、字节和完成事实，不从 pass 名字推导行为 |

默认离线编译环境是source已记录的CANN set_env.sh后，在PATH中可解析到bisheng（以shutil.which/command -v与实际可执行文件身份确认）；set_env.sh不设置BISHENG_HOME。手工PATH或显式BISHENG_HOME为等效发现途径，不能把“是否设置BISHENG_HOME”当通过判据；禁止依赖target=auto的设备探测。同一源码的两种相反配置已观察到Ascend C差异，未知键硬失败，但最小kernel省略reuse键无字节差异不能证明该键未消费，独立效应需多buffer案例。JITKernel的n_regs/n_spills/n_max_threads=None、resource_usage={}、latency=None一律不作资源/性能证据。
换机、换env或编译器/库/目标配置变化后，旧编译与S0环境判据默认不可复用：先重建工具链fingerprint、重新离线compile与静态/事件核对，再生成该环境的新artifact；保留旧记录，不把其PASS转签。路径相同不证明环境相同，异机缓存须另证target_arch与host ABI/库兼容；设备执行资格仍由该机S3验证。
每项事实带 value/source/status，来源可为固定源码、编译观察或设备验证；候选所需但尚未确定的容量和机制保持未知，该候选在分析边界上不可构成完整问题。当前输入包缺失的参考实现与计划材料在原始只读包中完整存在，可借鉴算法并迁入现有编译段；其 CPU 模型通过不代表 TF 接线或设备验收，见[输入完整性调查](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/INPUT_FACTS.md:5>)。
整个交付只支持一套 typed schema 与一条原生主链；不实现 text twin、按实际 trip count 静态展开、算子直调 tilelang.compile、metadata-only 安装、历史版本分派或另一套 JSON 编译器。有限模板内部的小常量展开属于指令 lowering，运行时循环和调度权威仍由实际 TF 程序及唯一 ResolvedPlan 决定。
本文为只读调查后的开发计划，没有调用设备或将未来阶段标记为完成。
