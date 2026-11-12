# v3 算子、候选生成与真实设备验收规格

本文使用固定基线 `TileFoundry@f4d6fe11769974bb31396acf890acc710749de5c`。它规定下一轮实现，不声称下文新接口已存在。原 `attention_residual.tar`、`mla.tar` 中的 `golden.py`、测试入口、输入生成器和用例配置仍是数值合同。`optimization/` 中的历史程序只可用于理解旧实验。三 kernel MLA 不进入本轮架构、主路径或最优基准。

## 1. 目标与入口

每次 operator 调用应执行一个逻辑计算 kernel。MLA 的一个 AIC 与配对 AIV 程序可以属于同一次逻辑调用。host 应只校验元数据、选择 specialization、分配输出、绑定已有 tensor 并调用一次 kernel。host 不应执行额外的 concat、transpose、cast、matmul、softmax 或 residual kernel。

生产链应为：原生 Scheduled HIR → 正常 TF 分析与求解 → 唯一 ResolvedPlan → 已安装为实际入口的最终 TIR → 显式 TL → binary → 原测试。算子构造器应创建真实 native IR 节点。算子配置 dataclass 可描述搜索参数，但不替代 HIR。JSON 应只保存 specialization、产物标识、诊断和结果。

完整入口放在 `attnres/build.py`、`mla/build.py`、`runtime/operators.py`。命名可随主工作包统一，职责应保持：

```python
# 接口规格；实现应返回项目已有的 Module/Expr/Buffer/Function 类型。
def build_attention_residual_hir(case, candidate, target) -> Module: ...
def build_mla_hir(case, candidate, target) -> Module: ...

def compile_operator(case, candidate, target):
    module = native_builder(case, candidate, target)
    compiled = normal_tf_compile_entry(module, target)
    # compiled.final_tir.entry_function 是实际安装的新入口。
    # compiled.plan 是该入口执行的唯一计划。
    verify_candidate_knobs_applied(case, candidate, compiled)
    return compiled

def attention_residual(v_stack, x, w, gamma, alpha, epsilon=1e-6):
    case = validate_metadata_only(...)
    artifact = dispatch.resolve(case)
    out = allocate_output(case)
    artifact.invoke_once(v_stack, x, w, gamma, out, alpha, epsilon)
    return out

def mla(q_nope, q_rope, k_nope, k_rope, v, numKVHeads=1,
        scaleValue=-1.0, inputLayout="BSND", is_causal=False):
    case = validate_metadata_only(...)
    artifact = dispatch.resolve(case)
    out = allocate_output(case)
    artifact.invoke_once(q_nope, q_rope, k_nope, k_rope, v, out)
    return out
```

`normal_tf_compile_entry` 是本文的职责占位，不是新增平行编译器入口。实现时应接到项目已有 analyze/finalize/emit 或其正式统一入口。不得在算子文件里保留直连 `tilelang.compile` 的验收旁路。

## 2. 当前实现应如何迁移

| 当前内容 | 可保留内容 | 应改变的内容 |
|---|---|---|
| `attnres/model.py` | 数学、ABI、形状校验、score policy 概念 | family 不再仅由 CN 命名；它应描述真实驻留方式 |
| `attnres/hir.py` | LOAD/COMPUTE/STORE 的概念与 V 最后消费者说明 | 自建 Task graph 和按全部 group/N 展开不进生产；替换为 native For/VF/DMA 节点 |
| `attnres/twin.py` | 已验证的指令用法、RMS/softmax 算式、真实 D 地址形式 | 不再直接从 spec 拼文本；最终 TIR 决定发射顺序、guard、slot、同步 |
| `attnres/kernel_persist.py` | 旧复现现象和反例 | γ/w 每轮重读只是旧路径绕过方法；重验正确驻留，不把它固化为目标规则 |
| `mla/model.py`、`sharding.py` | GQA、scale、right-aligned causal、列片覆盖检查 | row mapping、列片和 grid 应出现在实际 native IR 中 |
| `mla/kernel.py` | panel、stride、L0C→UB、UB→L1、P/PV 的指令经验 | 固定版本和固定 stage 应迁入候选策略；去除正式路线的直接 TL 自动调度 |
| `mla/rescale.py` | 独立数学、epoch 负例 | 新 native state effect 和 completion 应消费同一规则；CPU 模型不算设备实现 |
| `search/process.py`、`harness.py` | 子进程 deadline、原始样本统计、预算结构 | 增加真实 builder/原 runner/profile 适配器；设备 ranking 不接受模拟数据 |

本轮已确认的新反例应先进入回归。`attnres/twin.py` 的 factored score 漏 RMS。`token_rows/d_chunk/v_versions` 改变后可能生成完全相同的 TL。MLA 列分片增加 jobs 后，wrapper 仍使用未分片 cores。详细复现见工作包中的 `evidence/reproduce_operator_counterexamples.py` 与 `evidence/operator_counterexamples.json`。这些是 CPU 源码与数学反例，不是 NPU 结果。

## 3. attention_residual 数值合同

输入 `V[N,B,T,D]`、`x[B,T,D]`、`w[D]`、`gamma[D]`。四输入 dtype 一致，为 BF16 或 FP16。令 token 行 `r=b*T+t`，`M=B*T`。

\[
q_{n,r}=\sum_d V_{n,r,d}^2,\quad
\rho_{n,r}=\sqrt{q_{n,r}/D+\epsilon}
\]

\[
s_{n,r}=\frac{1}{\sqrt D}\sum_d
  \left(\left(V_{n,r,d}/\rho_{n,r}\right)\gamma_d\right)w_d
\]

\[
p_{n,r}=\operatorname{softmax}_n(s),\quad
y_{r,d}=\operatorname{cast}_{x.dtype}
\left((1-\alpha)x_{r,d}+\alpha\sum_n p_{n,r}V_{n,r,d}\right)
\]

golden 中间计算为 FP32；数值路径变换是否可用，由原比较器决定。加权求和使用原 V。RMS 与 score 归约沿 D，softmax 沿 N。实际 D 用于除法与 sqrt(D)，padding 不进入数学维度。

描述文件的合同包括：`2<=N<=64`、`256<=D<=8192`、`0.05<=alpha<=0.95`、`1e-8<=epsilon<=1e-3`、`N*B*T*D<=2**28`。不得以 alpha=0 或 N=1 跳过一个输入来替代合法域验收。原 20 例覆盖 D=2047、1000 等尾部；16/17 例 V 各为约 512 MiB。

### 3.1 score policy 应产生不同且正确的 IR

`literal_fp32` 保持除 RMS、乘 gamma、乘 w 的语义顺序。向量归约次序可以变化，但不可偷偷替换为另一 score policy。

`factored_gw` 可在同一 kernel 内计算 `gw=gamma*w`，然后：

```python
q = reduce_sum(v * v)
dot = reduce_sum(v * gw)
score = dot / sqrt(q / D + epsilon) / sqrt(D)
```

上述 RMS 因子不可遗漏。当前 `attnres/twin.py:108–114,218` 就遗漏了它。固定测试应使用 `N=2,D=256,V0=1,V1=0.5,gamma=w=1,x=0.25,alpha=0.5,epsilon=1e-6`；两个 V 行的归一化 score 几乎相同。漏 RMS 会改变 softmax 与输出，无法被宽泛的随机形状测试可靠定位。

### 3.2 最小 native 模板

一个 AIV program 处理 R 个 token 行。最初可用 R=1、CN=1、V1、串行正确模板闭合完整原 20 例；之后从同一构造器增加 pipeline 和性能族。

| Unit | 域/执行资源 | native body 与读写 | 完成和寿命 |
|---|---|---|---|
| `LOAD_WEIGHTS` | AIV/MTE2，转换可另为 VF | 读取 gamma/w，写 UB；factored 可另写 gw | 常驻时活到该 program 最后一个 token group 的 score 完成 |
| `INIT_GROUP` | AIV/完整 VF | 为有效 token 行写 m、l、U 初值；无效行写安全初值 | 每个 token group 应重新初始化 |
| `LOAD_V(k)` | AIV/MTE2 | 真实 GM view → `Vslot[CN,R,Dpad]`，无效区清零 | 写完成后才 READY；复用需等所有 score/weighted consumers |
| `SCORE_BLOCK(k)` | AIV/完整 VF | 读取 V 与 weights；写 scores、RMS scratch | V 此时仍不能释放 |
| `MERGE_BLOCK(k)` | AIV/完整 VF | scores 求 block state；读原 V；更新 m/l/U | V 的最后读通常在此；m/l/U 为 distance=1 recurrence |
| `LOAD_X(g)` | AIV/MTE2 | 真实 x view → x UB | 可按容量与阶段预取；guard 应与该 group 一致 |
| `FINALIZE(g)` | AIV/完整 VF | 读取最后 U/l 与 x；写输出 UB | alpha/epsilon 按实际 ABI 绑定 |
| `STORE_Y(g)` | AIV/MTE3 | 输出 UB → 真实 GM D 列 | output UB FREE 发生在 MTE3 读完成；退出需等待必要写完成 |

`SCORE_BLOCK+MERGE_BLOCK` 最初可作为一个完整 VF unit，以降低跨 VF scratch 与 scalar-state 难度。后续若拆分，应让 captures、scalar 状态和 buffer effects 可见，不可把每条 VF 指令独立丢给外层 solver。

外层 token group loop 应保留原生循环：

```python
for group in serial(program_id, ceildiv(M, R), launched_programs):
    row0 = group * R
    # row validity: row0 + r < M
    initialize(m, l, U)
    for k in scheduled_loop(ceildiv(N, CN)):
        # n = k*CN+c; validity n<N
        load_and_merge(k)
    finalize_and_store(row0)
```

这里的 `scheduled_loop` 表示 native For 上的作者策略；不得用 Python `range(真实tripcount)` 创建所有 token/N 的 IR。固定 shape specialization 也应保留有限 loop body。只有固定、很小的指令模板局部展开可由编译器决定。

### 3.3 三个主要性能族的真实区别

| 策略族 | 工作方式 | GM V 访问 | 驻留量 |
|---|---|---|---|
| `online` | CN=1，score 后立即更新 running state | 每个 V 元素一次 GM 读；weighted 阶段从 UB 再读 | 一个或多个 ring slot |
| `blocked_online` | 1<CN<N，保存一个 CN block 的 V 和 score，合并一次 block state | 能驻留时每个元素一次 GM 读 | `versions*CN*R*Dpad*element_bytes` 加 state |
| `resident_n` | CN=N，全部 N 的 V 与 score 在该 token group 内驻留，做一次 softmax/weighted sum | 每个元素一次 GM 读 | 整个 N block；容量可限制适用 shape |

另可保留 `blocked_reread` 策略：只保留 score 或重算 score，分两遍读 GM，换取更低 UB 占用。它是第四个实现选择。`CN=N` 的两遍重读版本不能标为 `resident_n`。是否有竞争力由设备计时决定，不预判优劣。

在线更新可使用下式，避免完整 N 分数驻留：

\[
m'=\max(m,m_b),\ a=e^{m-m'},\ b=e^{m_b-m'},
\quad l'=al+bl_b,\quad U'=aU+bU_b
\]

首次有效 block 应使用显式 empty-state 路径，或证明安全的初值逻辑。不能让无效 token 行执行 `-inf-(-inf)`。尾部 N 的 score 应为负无穷或等价零权重；尾部 D 应在 load/算式前屏蔽，不能越界访问后再 mask。

### 3.4 每个 AR 搜索 knob 的兑现位置

| knob | 应改变的实际实现 | 不能接受的情况 |
|---|---|---|
| `token_rows=R` | group loop 步长、row view、V/state/output 形状、row guard | 只改资源报告，正文仍一 token |
| `block_n=CN` | load block、score/block state、softmax 合并频率与 live V | 仅在注释中改 family |
| `d_chunk` | VF 的向量遍历、局部归约、scratch 生命周期 | 只改配置 hash |
| `v_versions` | unique plan 的 versions、slot、FREE、真实 allocation 和 TIR 地址 | 配置写 V2、后端只有一个 buffer |
| stage/issue order | native schedule attrs → solver → 物理 loop 中的发射顺序 | 独立解析计划与 kernel 保持固定 serial |
| `score_mode` | literal 或带完整 RMS 因子的 factored 算式 | factored 标记下继续 literal 或漏 RMS |
| `vf_mode` | 完整 SimdVF/SimtVF native body 与目标指令 | 函数名不同但同一正文 |
| prefetch | X 或下一 group 首 V 的真实 task、guard、版本与 release | 提前发射却无对应 lifetime |
| weights residency | 真实 UB allocation 活跃区间及跨 group 读依赖 | 把旧重载绕过办法当硬件应条件 |

AR 初始搜索 R∈{1,2,4,8,16}，CN∈{1,2,4,8,16,32,64,N} 取合法子集；Dchunk 取目标支持的向量 chunk；V versions 从 1、2 开始，再考虑 3。实际 layout、alignment、mask 和 spills 决定资源；Simt 的 power-of-two 限制不能自动套用到新的 Simd 模板。

### 3.5 AR 必要定向验收

- 同一符号 token loop 实际处理多轮 group，γ/w 只读一次的版本与每轮重载版本数值一致；核别名与同步原因。
- D=2047、1000、5120、7168，以及 N=3/6/12/24/32 和 M 非 R 倍数。
- V2/V3 在 tripcount 小于 pipeline height 时正确，N/UB slot 不错位。
- 修改实际 operand、alpha、RMS divisor、GM offset、last consumer release，应由结构核对或原数值测试拒绝。
- literal 与 factored 分别经过原比较器；family 的实际 GM 读流量和驻留 buffer 与标签相符。

## 4. MLA 数值合同

`q_nope,q_rope,k_nope,k_rope,v` 采用同 dtype、同 layout。`V==K_nope` 是数值合同，原 `get_input` 通过 clone 建立它，不能要求同一指针。kernel 可直接复用 K_nope 内容；host 不应为了校验值等价再运行比较 kernel。

令 `G=Nq/Nkv`、`hkv=hq//G`、`Dn=d_nope`、`Dr=d_rope`：

\[
Z_{i,j,hq}=scale\left(
Q^n_{i,hq}\cdot K^n_{j,hkv}+
Q^r_{i,hq}\cdot K^r_{j,hkv}\right)
\]

`scaleValue<=0` 时 `scale=1/sqrt(Dn+Dr)`。rope dot 应在全部 nope panels 累加后加入一次。causal 时 `j>i+Skv-S` 的 score 被屏蔽。输出为 `softmax(Z) @ V`，dtype 与输入一致、layout 保持不变。

原 MLA 20 例包括 Dn=448/512、Dr=64、BF16/FP16、BSND/BNSD、causal/noncausal、decode、MTP、prefill、S>Skv 的非因果 cross attention。原用例 Nkv 均为 1；扩展测试仍应覆盖 Nkv>1 与显式正 scale。case11 的实际 Skv=1024，不能使用备注中的 4096。case15 为 BNSD B2,Nq32,S64,Skv512；case18 为 BSND B4,S2,Nq32,Skv512。

### 4.1 统一行映射

一个 MIX program 只处理一个 `(batch,kv_head)` 范围，行 index 可融合 query position 和 GQA group。

| layout | 连续逻辑行 r | 反解 query position i / group g |
|---|---|---|
| BSND | `r=i*G+g` | `i=r//G; g=r%G` |
| BNSD | `r=g*S+i` | `g=r//S; i=r%S` |

`hq=hkv*G+g`。causal mask 应使用反解后的真实 i。选择不同 row mapping 是空间策略，可以为访存和 causal 跳块比较多种实现；实际 TIR 地址与 guard 应采用同一个映射。

所有公共输入仍按原 layout 访问。metadata view 可用于绑定，但它不能触发数据拷贝。原 wrapper 的 `as_strided` 经验可复用。应校验全部输入的 shape/dtype/device/layout 元数据，不只检查 q_nope dtype。

### 4.2 canonical online 状态

每个 query 行保留 FP32 `m,l,U`：

\[
m'=\max(m,\max Z_b),\quad a=e^{m-m'},\quad P=e^{Z_b-m'}
\]
\[
l'=a l+\sum P,\qquad U'=aU+PV
\]

`m/l` 与 `U` 是不同 recurrence。前者不必等待所有输出列的 PV；后者应等待自己的旧 U 与当前 PV。alpha/P 在全部相关消费者完成前不可被下一 block 覆盖。最终 `Y=U/l`；无效 padded query 行使用安全 denominator 并不写 GM。

默认计划应保持 QK accumulator、score、softmax/state 为 FP32；Cube 的 P operand 可转换为目标支持的低精度。当前 TL 代码中 score/scale 后两次低精度舍入不是原比较器对所有实现的硬要求。若保留此路径，应命名为独立 precision policy，并与原 runner 比较。

`p_single` 的 numerator 使用 cast 后 P、denominator 使用 FP32 P。若原比较器不接受该误差，可用 `p_hi_lo`：

```python
P_hi = cast(P)
P_lo = cast(P - fp32(P_hi))
contribution = mma(P_hi, V) + mma(P_lo, V)
# denominator 仍为 sum(P)，不是 sum(P_hi + P_lo)。
```

两次 MMA、两个 P component、额外 staging 与最后消费者应进入原生 IR/effects、solver 和资源计费。

### 4.3 最小 native MIX 模板

| Unit | 域/执行资源 | native body | 关键完成合同 |
|---|---|---|---|
| `LOAD_Q` | AIC/MTE2 | GM Qnope/Qrope → L1 | 一个 query job 内复用 |
| `LOAD_KV(k)` | AIC/MTE2 | GM Knope/Krope → L1 ring | READY 后 QK/PV operand staging 才能读 |
| `QK_OPERANDS(k,panel)` | AIC/MTE1 | Q/K L1 → L0A/B | 不可提前复用仍被 Cube 读取的 operand slot |
| `MMA_QK(k,panel)` | AIC/Cube | FP32 score L0C 累加 | 首 nope panel clear；rope 加一次 |
| `SCORE_TO_UB(k)` | AIC/FIX，配对 AIV 目标 | FP32 L0C → FP32 UB，按 row 或 col split | dual copy 同 dtype；完成后 AIV 读取 |
| `SOFTMAX_PACK(k)` | AIV/完整 VF | mask/scale/max/exp/l/alpha；P cast | 更新 m/l；写 P 与当前 alpha；对称 guard |
| `PACK_NZ(k)` | AIV/完整 VF | P 的 ND UB view → 已声明物理 pitch 的 NZ UB view | scratch、cast 与读写效果先进入 TF 分析；发布 V→MTE3 完成 |
| `P_TO_L1(k)` | AIV/MTE3 | 已 NZ 的 UB P → L1 | 等待 PACK_NZ 完成；P NZ UB FREE 在此读完成；两 SID 产物均可见后 AIC 使用 |
| `PV_OPERANDS(k,col,component)` | AIC/MTE1 | P、复用 K 的 V 列片 → L0A/B | L1 K/P 的 FREE 取真正最后一次 MTE1 读完成 |
| `MMA_PV(k,col,component)` | AIC/Cube | FP32 PV L0C | 所有列、hi/lo 均计入 effects |
| `PV_TO_UB(k,col)` | AIC/FIX | FP32 L0C → FP32 UB | 完成后 U update 才能读 |
| `UPDATE_U(k,col)` | AIV/完整 VF | `U=a*U+PV` | 当前 alpha 活到所有列消费者结束 |
| `NORMALIZE_STORE(job)` | AIV/VF + MTE3 | U/l、cast、按真实 row/col 写出 | output staging 活到 DMA 读完成；kernel exit 完成写出 |

外层 solver 不应盲目把 panel 内每条指令作为任务。可把固定小 panel loop 封装成具有已验证内部顺序和 resource/effect 合同的模板；模板最终仍生成真实 native 节点。若 composite 横跨 MTE1/Cube/FIX，应显式声明内部合同，不能假标成一个单 pipe unit。

UB→L1 的 packing 遵守 [04 §6.5 的后端 refinement 合同](04_completion_storage_backend.md#65-关闭-autoschedule-后的剩余-tl-lowering)：首版显式形成 `PACK_NZ` whole VF 与 `P_TO_L1` MTE3 两个任务，并在其间生成真实 V→MTE3 同步。`PACK_NZ` 可在已验证的同一 VF 内与 softmax/cast 合并；跨 VF/MTE3 的 composite 应证明内部完成合同。scratch、physical pitch、versions 和同步资源应在 TF 调度前可见。不能将 raw ND→L1 的 `T.copy` 当成只有 MTE3 效果的单 pipe task，再依赖关闭 AutoSchedule 后仍运行的 TL 隐式转换补齐同步。

canonical 初版可采用保守 serial 计划。它的完成标准仍是完整两个算子经正常入口成功。然后逐步开放 stage/phase/versions 和 overlap；不能把 serial 投影的受限能力永久当成四维策略支持。

### 4.4 双向复用应按实际 buffer 区分

K_L1 同时用于 QK 与 PV。它的最后读可能是最后一个 V panel 的 `MTE1` 完成，而不必等到整个 PV MMA 完成。对应的 L0B slot 则应活到 Cube 读完成。P_L1 的最后读同样是全部相关 component/column 的 operand staging 完成。P_UB 活到 UB→L1 读完成。把这些概念都粗略写成 “PV 完成” 虽可保守正确，但会限制版本复用和 pipeline 性能。

alpha 与 U 的消费者是 AIV VF，其寿命不能从仅有 m/l recurrence 推断。solver、completion、storage 应基于同一真实 buffer/effect 集合。

### 4.5 原 shape 与尾部支持

第一步应支持原 20 shapes 的全部 layout/dtype/dimension 组合；下一步加入真实 KV tail、query tail 和全无效 padded rows。当前模板不支持 BC64/BQ16，不代表 Ascend950PR 不支持这些选择。区分以下三种结论：

- `TARGET_UNSUPPORTED`：有已核硬件或工具链约束与对应失败证据。
- `TEMPLATE_MISSING`：当前 native 模板尚未实现，不属于已探索但不可行路线。
- `CANDIDATE_ILLEGAL`：这个 shape 与已实现指令/资源/guard 的组合不可用。

KV tail 应用固定物理 BC allocation 和有效元素数表示。先用合法方式清零无效 operand，再只搬运 GM 中存在的数据；score 的无效 KV 位置置负无穷，softmax 权重为 0。不得用完整越界 DMA 加事后 mask。

对于 Dn448，可用 64 panel 覆盖 7 块；也可为 128/256 物理 panel 填充尾部，再按真实 D 归约/写出。后一种可能减少某些调用但增加 padding 工作量，应测量。最初可保留 64 路径保证正确，随后作为真实性能候选扩展。

### 4.6 输出列分片仍是一个 kernel

推荐按输出宽度 `BD_shard` 定义 program 列区间，而不是只允许等整除的 shard count：

```python
shard_count = ceildiv(Dn, BD_shard)
job_count = B * Nkv * ceildiv(G*S, BQ) * shard_count
launched_programs = choose_grid(job_count, target.mixed_group_count, candidate)

for job in serial(program_id, job_count, launched_programs):
    shard = job % shard_count
    row_job = job // shard_count
    col0 = shard * BD_shard
    valid_cols = min(BD_shard, Dn-col0)
    # 完整 QK/softmax；只计算/写本 shard 输出列。
```

Dn448、BD_shard128 对应四片，末片 64 个有效列。物理 allocation 可以统一 128，padding lane 不读越界 GM、不参与有效输出。各片独立完成完整 softmax，不执行跨 program 合并，不产生第二 kernel。

grid 应在 shard 后计算。当前 `mla/kernel.py:393` 先按未分片 jobs 限制 cores，因此 B1/S1/Nq128/BM32/shards4 的 jobs 从 4 变 16，实际 launch grid 仍 4；这无法验证增加独立并行度的价值。

列分片会重复 QK/softmax 和部分 K 流量。它应作为小 decode 重点比较项，不能普遍强制。资源减小与并行度提升是否抵消重复工作，应用设备样本说明。

### 4.7 deferred rescale 是最终必测路线

先用 canonical 闭合全部原用例，再实现 deferred。CPU epoch 模型只提供不变量，生产实现应在 native effects/completion 中表达它。

每行维护 anchor `a`、denominator `l`、AIV prefix `U_ub`，AIC 保留 `pending_l0c`。在同一 anchor 下：

\[
U_{total}=U_{ub}+pending_{l0c}
\]

如果新 block 不要求提高 anchor，PV 直接向 pending 累加，可以省去该 block 的 PV→UB 与 rescale。若需要变化，应严格执行：

1. 等旧 anchor 的全部 PV 完成，覆盖每列、每个 hi/lo component。
2. 把 pending 搬到 UB；与 prefix 相加。
3. 每行乘 `exp(a_old-a_new)`；未改变 anchor 的行 factor=1；相同 factor 更新 denominator。
4. clear pending；等旧 epoch predicate/factor 的全部消费者结束。
5. 发布新的 anchor epoch；此后才允许新 epoch 的 PV 累加。

两个 AIV 的局部判定应形成同一个 MIX group 判定。SS 可存放组内小标量及 generation，但不提供跨 program barrier；不能据此实现跨 program split-KV 合并。本轮不增加 split-KV 或多 kernel merge。

softmax 的 scale、anchor、theta 应采用一致单位。theta=0 的 deferred 是首先需要验证的版本；再比较正 theta。原用例 YAML 的 V∈[-1,1] 可以提供这些固定用例的静态幅值界。若计算防溢出界，应包含整个未 drain epoch 的有效 KV 数及所有累计量，而不只一个 BC tile。`V_abs_max=8` 这种默认常量不能单独成为泛化接口域的合法性证明。目标是通过原比较器并获得实际收益，不以经验阈值冒充证明。

必要反例应直接打到 native effects/最终 TIR：旧 anchor 的某个列或 low component PV 迟到、factor 提前覆盖、不同 SID 走不同分支、最后 pending 未 drain。任一情况应被拒绝或使独立数值测试失败。只修改 CPU 模型的 gate 不能完成 deferred 验收。

## 5. 搜索应连接真实候选

### 5.1 四层判断

候选依次经过：模板可用性 → 正常 native 编译与最终资源核对 → 原数值比较与单逻辑 launch 检查 → 设备计时。粗略 cost/resource 估计只用于排序或识别已证明的资源下界，不得用错误的手写估算淘汰可实现候选。

当前资源估算的具体问题是：重复计入 Qrope、以共享 Cube 推断 L0A/B 必然不重叠、为纯 AIV AR 添加不存在的 L1 staging。新代码应从生成的真实 IR、ResolvedPlan 和 StoragePlan 自动得到 buffer shape/versions/alignment/owner/lifetime，最终把 TL 后端增加的 staging 与 spill 纳入实际容量。

若一个预筛条件确实是已证明的硬下界，例如不可分割 live buffer 本身已超过目标容量，可提前拒绝。拒绝报告应包含该 buffer 与对应计划条件。其余估计只影响搜索顺序。

### 5.2 配置到产物的最小适配器

```python
def evaluate_candidate(case, candidate, budget):
    capability = builder_capability(case, candidate)
    if capability.template_missing:
        return result("TEMPLATE_MISSING")
    if capability.proven_target_illegal:
        return result("TARGET_UNSUPPORTED", evidence=capability.evidence)

    artifact = compile_in_clean_tf_worker(case, candidate, budget)
    # 必须是正常编译入口，下面读取真实结果。
    assert artifact.installed_entry == artifact.final_tir.entry_function
    assert artifact.plan_complete
    verify_knob_effects(candidate, artifact)
    verify_final_resource_use(artifact)
    semantic_key = canonical_native_plan_key(artifact)
    if already_measured(case.key, semantic_key):
        return result("DUPLICATE_REALIZATION")

    numeric = run_original_case_in_device_worker(case, artifact, budget)
    if not numeric.passed:
        return result("NUMERIC_FAILED", raw=numeric)
    check_single_logical_launch_and_plan_binding(artifact)
    samples = calibrated_kernel_samples(artifact, budget)
    return measured_result(case, candidate, artifact, samples)
```

这些是职责接口。它们应适配已存在的 `search/process.py::WorkerCommand`、`run_process`、`Deadline`，不再造一套超时框架。CPU compile worker 不导入 torch_npu；设备 worker 不执行 TF/ORTools memory analyze。每个阶段应绑定同一 plan/TIR/TL/binary 身份。

配置 hash 只在生成前去重。正式候选多样性应按实际实现判断。canonical plan key 应忽略纯 flag 重编号、绝对地址重定位、注释、报告名称；保留 loops、真实 operands、precision、core、stage/phase/II、versions、guard、layout、storage shape 与策略差异。两个不同 knob 字典若解析成同一个计划，只计一个候选。

### 5.3 应兑现的搜索维度

AR：R、CN、真实驻留/重读、Dchunk、VF mode、score policy、versions、stage/issue order、prefetch、grid/persistent group mapping。

MLA：BQ、BC、QK/PV panel、row mapping、KV/P/score 等独立版本、stage/issue order、canonical/deferred、theta、P single/hi-lo、输出列片宽度、K=V 复用、grid。各 knob 应能在真实 IR、计划和实际分配中找到作用。如果尚未有 backend，就标 template missing，不能把模型参数作为已测试方案。

### 5.4 初始覆盖与预算

保留 v2 的初始覆盖：每算子至少 8 个代表，每个代表至少 8 个实质不同且数值正确的设备候选；每个原 case 至少 3 个可行竞争候选；代表 winner 应有局部邻域验证。初始每 case 候选上限 40，分为最多 12 seed、20 neighbor、8 advanced；设置共享编译/运行/总时间 deadline。

初始 AR 代表为 2、4、8、13、14、17、18、20；覆盖小 token、非对齐 D、FP16、大流量、宽 D、epsilon 和大 N。初始 MLA 代表为 2、3、6、7、9、14、15、18；覆盖两种 layout/dtype、Dn448、cross、单 batch decode、大 batch、MTP 与 G32。这与 `templates/search_budget.example.json` 一致。P0 可按实际 case_matrix 调整或增加代表，记录覆盖理由；不能减少关键维度覆盖。

阶段实施允许先用小预算跑通真实候选评估。最终计数仍使用上述覆盖。预算耗尽应产出明确未完成工作与剩余矩阵，而不是把缺代码、缺设备或未实现算法族改标不可行。若已实现合法空间确实不足 8 个不同方案，应提交空间枚举和实际约束证据，由 Lead 在最终报告解释，不能重命名重复方案凑数。

`resident_n` 不要求所有大 shape 都可用；它应在能真实驻留的代表上参与比较，并为其余 shape 给出最终资源拒绝原因。deferred/列片应在合适 MLA 代表上真实参与；不要求每个 shape 都选它们。

### 5.5 计时和 winner

设备 worker 应使用目标环境已验证的事件计时或 profiler；同 stream 对一次实际 operator kernel 的开始/结束计时，按计时 API 的单位换算为 μs。compile、输入生成、oracle、host allocation、warmup 不计入 kernel 时间。初始化、同步与 profiler 开销应单列。

初始使用 10 次 warmup、每批 50 次、3 批样本。样本不足、来源不明或 simulated clock 不进入设备 ranking。短 kernel 如需一次计时区间内重复多次，应记录重复数，确保分母与实际调用数相同，并保持一次 operator invocation 的单逻辑 launch 合同。

候选比较应交错或随机顺序复测，避免固定顺序温度/缓存偏差。输出 median、p10/p90、原始样本、样本数与 target/toolchain 身份。noise 内相同应视为并列，再按真实资源、稳定性和确定性规则选择。当前 `rank` 仅写了此规则而没有实现 footprint tie-break，应补齐。

通过原比较器的最快候选只能称为当前搜索范围中的最佳结果。不可仅凭 8 个候选或一个理论带宽数字宣称全局最优。性能报告应说明当前瓶颈、为什么再改下一个 knob，以及邻域探索的收益是否超过测量噪声。

## 6. 原 40 例验收与阶段顺序

### 6.1 权威 runner

应使用原 `test_attention_residual.py::main(operator_api)` 与 `test_mla.py::main(operator_api)`，通过原 `cannbench_contract.Harness` 和真实 `npu_compare` 比较器。不得用手写 20 例清单或直接比较某个历史 kernel 替代。

case key 为 `level4/attention_residual_1..20`、`level4/mla_1..20`，seed 沿用原 helper。MLA 应让原 `get_input` 建立 V 的合法值合同，再向 oracle/candidate 提供等价输入。最终验收默认禁用旧 oracle cache；若为了长搜索复用 cache，应绑定输入/golden/helper/comparator/seed 的确切身份，最终完整复验仍应从干净身份开始。

数值通过、计划对应、资源可执行、单逻辑 launch、性能样本是五种不同结果。一个结果不能替代另一个。若原比较器不可用，数值 gate 为环境阻断；不得换阈值或用 CPU 模型签通过。

### 6.2 增量顺序

| 步骤 | 交付 | 正常入口的最小验收 |
|---|---|---|
| O0：保留已知反例 | 新反例回归；真实候选参数兑现检查 | 漏 RMS、ignored knobs、错误 shard grid 均能被指定测试检测 |
| O1：完整 AR canonical | native literal CN1、R1、serial；完整 wrapper | 原 20 例全部通过，实际 TIR/显式 TL/单调用对应 |
| O2：AR pipeline/驻留 | symbolic loop、V2、持久 group、blocked/resident；合法 tails | 多轮 group 和短 loop 无错，所有族经过同一入口；完整 20 重跑 |
| O3：MLA canonical | native MIX、QK→softmax→P→PV→normalize；layout/dtype/448 闭合 | 原 20 例全部通过；单逻辑调用；不在 GM 写完整 scores/P |
| O4：MLA pipeline/列片 | 独立 KV/P 版本、行映射、尾部、shard 后 grid | Nkv>1/scale/tails 定向测试和原 20 复验；列片 actual jobs/grid 有变化 |
| O5：MLA deferred/precision | epoch state effect、全 producer completion、theta 与 hi/lo | 注入旧 PV/factor 错误被拒绝；真实设备原比较器与收益比较 |
| O6：搜索与最终 dispatch | 有效候选适配、设备 profile、winner 配置 | 8×8 代表覆盖、每 case3 竞争候选、最终 dispatch 原 40 例完整 PASS |

O1/O3 的 serial baseline 是迅速建立完整链的步骤，不能替代最终四维调度支持。AR 与 MLA 的前端模板可由不同 agent 并行编写；共用 compiler/backend 的接口由 Lead 冻结，所有 NPU 操作由持锁执行者串行完成。不能让多个 agent 各造一套 solver 或直接 TL 调度路线。

## 7. 来源与证据界限

当前实现审计来源：

- [当前交接说明](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/REVIEW_HANDOFF.md)
- [AR 旧 HIR](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/attnres/hir.py)
- [AR 旧 twin 与 checker](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/attnres/twin.py)
- [MLA 现有直接 TL 模板](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/mla/kernel.py)
- [搜索 generator](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/search/generator.py)、[filter](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/search/filter.py)、[harness](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/search/harness.py)
- [TileLang dual_copy 的实际合同](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/tilelang/ascend/language/copy_op.py)

数值语义来自工作包保留的原附件 fixtures。本文新算法、接口、迁移方式属于设计要求。历史设备结果仍以对应原始日志和固定源码为界；本轮 CPU 反例只验证数学差异与旧 checker 行为，没有提供新的 NPU 性能或完整算子通过证据。

## 8. 可直接运行的核心代码与本规格的关系

工作包保留了独立 CPU 核心实现，便于本地团队先理解和移植算式。它们不调用 NPU，也不替代正式编译链。

| 文件 | 对应本规格 | 应如何使用 |
|---|---|---|
| `reference_core/online_softmax.py` | 3.3、4.2 的状态更新 | 作为独立数学参考，核验空 block、merge、原 V 聚合 |
| `reference_core/reference_ops.py` | 完整 ABI、BSND/BNSD、GQA、scale、causal | 用很小的 CPU 输入核验地址映射，不用它替代原 golden |
| `reference_core/attention_reference_core.py` | AR R/CN/literal/factored 与资源探索 | 可移植 FP32 算式；资源候选需接真实 native allocations |
| `reference_core/mla_online_reference.py` | canonical/deferred/列片数学 | 比较数值变化与正确定义；没有 Cube 指令性能含义 |
| `reference_core/rescale_epoch_reference.py` | 4.7 的尺度代次协议 | 移植全部 PV producer、factor/metadata 生命周期和负例 |
| `reference_core/completion_reference.py` | 3.2、4.4 的 READY/FREE | 理解全消费者完成；硬件事件仍由正式 completion 产生 |
| `reference_core/pipeline_reference.py` | 简化 stage/phase/ordinal | 只作小型数学参考；生产实现使用新工作包的 native pipeline 框架 |

在工作包根运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python reference_core/test_reference_core.py
PYTHONDONTWRITEBYTECODE=1 python reference_core/test_rescale_epoch.py
PYTHONDONTWRITEBYTECODE=1 python evidence/reproduce_operator_counterexamples.py --repo /path/to/TileFoundry_at_f4d6fe1
```

前两条本轮实际结果分别为 14 与 6 个测试通过，共 20 个 CPU 数学/协议测试。第三条是旧固定基线反例复现器，只接受 `f4d6fe1` checkout；在修复后的代码上应建立正常入口回归测试，而不是继续要求旧故障存在。

`cases/case_matrix.json` 的 40 行已经与包内 18 份原始源文件的 SHA256、YAML shapes/dtypes/attrs/value_range、case keys 与 seeds 逐项核对；结果在 `evidence/fixture_consistency.json`。这些文件和 `tools/run_contract.py` 可随包迁移，无需用户原来的本地目录。真实运行仍需要本地 Torch/Torch NPU/TileLang/CANN、PyYAML 和原 `npu_compare` 实现；缺比较器不等于可换成自定义 allclose。

`cases/supplemental_mla.json` 另给 4 个精简诊断，覆盖 Nkv=2/4、两种 layout/dtype、正 scale、真实 KV/query tail 和 D448 最后一列片。 P4/T21 先用四个 shape 的 unsharded canonical 验证 GQA、scale、layout 和尾部；P5/T24 再执行 D448 的 sharded 对照与末列片检查。P4 不依赖尚未实现的列分片。它们使用独立 `supplemental/` key，不更改、不计入原 40 例，也不替代原 20 MLA 例。全 masked 测试优先验证 padding row 的内部语义；不要求把违反公开 causal `S<=Skv` 域的输入当成正式支持范围。
