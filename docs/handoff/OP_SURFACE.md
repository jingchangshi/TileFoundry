# OP-SURFACE（task-38）— AR/MLA 需要的指令面 vs 现有 ISA/ops 表

只读调查（explorer-g1）。基线 HEAD=f341e47；emit.py、completion/plan.py、apply.py 等是 implementer 未提交的工作树文件，行号后续需复核。TL 侧事实来自 site-packages/tilelang 的实测枚举（记 TLA=<site-packages>/tilelang/ascend/language）。全程未 compile、未 import torch_npu、未触设备。

## 0. 现有面快照（实测枚举）

- TF 已注册 ascend_* 指令 **17 个**（ir/ops.py:231-607）：GmToUb/UbToGm/Vld/Vsts/Pset/Pset16/Vcvt/Vadd/Vadds/Vmuls/Vmul/Vmula/Vcadd/Vdup/Vdupv/Vsqrt/Vdiv。
- ISA 契约表 **16 条**（ir/isa.py:74-206）→ **ascend_pset16 有 Op 无契约**：走 emit 即在 codegen/emit.py:525-529 抛 TL_INSTRUCTION_UNREGISTERED。
- TileLang 发射面 TL_SURFACE **只有 6 条**（emit.py:73-80：gm_to_ub/ub_to_gm→copy、vld→simd.vld、pset→simd.pset、vadd→simd.vadd、vsts→simd.vsts）。其余 11 个已注册指令 emit 时 TL_NO_SURFACE（emit.py:408-412）。
- 计算 dtype 面：VECTOR_DTYPES=(f32,)（ops.py:51）、LOAD_DTYPES=(f32,bf16)（:56）、TL_DTYPES 仅 f32（emit.py:79）。执行域只承认 aiv：MemoryExecutionFacts(("aiv",))（tf_target.py:144-145），aic 仅作为 sibling topology 层（:106-115）；emit 对 aic 直接 TL_UNSUPPORTED_CORE（emit.py:422-424）。
- **同步事件无可发射路径**：sync.py:23-32 的 AivFlagSet/AivFlagWait/AivDrain 只经 tf_target.completion_ops()（:189-191）暴露；三者均无 ISA 契约、不在 TL_SURFACE、apply.py 只复制既有指令语句（apply.py:227）、completion/plan.py 不点名它们 → 事件发不出来。

## 1. attention_residual 步骤 → 指令面（spec 05 §3、§3.2 表 :101-111、§3.3 :131-146）

| AR 步骤 | 需要语义 | 现有 TF 指令 | 契约 | 发射面 | 判定 / 域 |
|---|---|---|---|---|---|
| LOAD_WEIGHTS γ/w→UB | GM→UB 搬运 | GmToUb | 有 | copy | 有；bf16/fp16 输入面缺（§4） |
| INIT_GROUP 写 m/l/U 初值 | 标量/向量初值 | Vdup/Vdupv/Vadds | 有 | 无 | **发射 MISSING**（AIV，仅缺声明） |
| LOAD_V(k) GM→Vslot、无效区清零 | 搬运+pad 清零 | GmToUb（清零无指令） | 部分 | copy | **清零/设 pad 值 MISSING**；TL 侧 ascend_set_copy_pad_value 已有（TLA/dma.py） |
| SCORE q=Σ v²、rho=sqrt(q/D+eps) | 沿 D 归约 sum、div、sqrt | Vcadd 仅成对 pairwise（isa.py:122-140 自述树形 UNSOURCED）；Vdiv/Vsqrt 有 | 有 | 无 | **沿 D 整块 reduce_sum MISSING**（AIV 归约域）；TL 有 reduce_sum/max/min（TLA/reduce_op.py） |
| SCORE dot=Σ(v·γ)·w / factored gw | 乘+归约 | Vmul/Vmuls/Vcadd | 有 | 无 | 乘有面；归约同上；gw 不新增指令 |
| score 归一化 /(sqrt D)、+eps | 常量乘加 | Vmuls/Vadds | 有 | 无 | **发射 MISSING** |
| softmax：m'=max、P=exp(s-m')、l=ΣP | max、exp | **无 vmax/vmin、无 vexp** | 无 | 无 | **MISSING（AIV 向量域，需真注册）**；TL 侧 vexp/vln/vmax/vmin/vmaxs/vmins 均已存在（TLA/simd.py 实测 78 条 v*/p*） |
| online 更新 U'=aU+bU_b（§3.3 :141） | 乘加累加 | Vmula（isa.py:92-99） | 有 | 无 | 语义可用；**发射 MISSING**（distance=1 recurrence 属 solver 侧，非指令缺项） |
| MERGE 读原 V、写 block state | 搬运+乘加 | Vld/Vmul/Vsts | 有 | vld/vsts（vmul 无） | **MISSING：vcvt 与 vmul 无面 → bf16 原 V 加权不可发射** |
| FINALIZE Y=cast_x((1-α)x+αΣpV)、U/l | 除法 + 通用 cast | Vdiv 有；cast 仅 Vcvt 的 bf16carrier↔f32 两签名（ops.py:342-380） | 部分 | 无 | **MISSING：f32→bf16/fp16 通用舍入 cast**（AIV 转换域） |
| STORE_Y UB→GM + FREE/退出 | MTE3 + 事件 | UbToGm | 有 | copy | 搬运有；**FREE/退出 drain 需事件 → §0** |
| 尾部 N 置 -inf、尾部 D 屏蔽 | guard/mask + 无穷常量 | Pset（Pset16 无契约）+ mask 操作数 | 部分 | pset | **-inf 常量面 MISSING**（作者常量声明，非新指令） |

## 2. MLA 步骤 → 指令面（spec §4.2-§4.5，canonical MIX 表 :229-243）

| MLA unit | 域/资源 | 现有 TF 面 | 判定 |
|---|---|---|---|
| LOAD_Q / LOAD_KV(k) GM→L1 | AIC/MTE2 | 无（GmToUb 只到 UB） | **MISSING（transfer，AIC 侧）**；TL 有 alloc_l1+copy |
| QK_OPERANDS L1→L0A/B | AIC/MTE1 | 无 | **MISSING（transfer）**；TL 有 alloc_l0a/l0b |
| MMA_QK / MMA_PV L0C FP32 累加 | AIC/Cube | **零面**：isa.py 无 cube 条目、ops.py 无 MMA/Gemm Op、tf_target 不承认 aic 执行、emit 主动拒绝 | **MISSING（Cube MMA 域，最大缺项）**；TL 侧 gemm/gemm_blockscaled 已有（TLA/gemm_op.py） |
| SCORE_TO_UB L0C→UB dual copy | AIC/FIX | 无（tf_target.py:56-66 只是事实登记） | **MISSING**；TL 有 dual_copy（TLA/copy_op.py） |
| SOFTMAX_PACK mask/scale/max/exp/l/α、P cast | AIV/VF | 同 AR：max/exp 缺、cast 仅 bf16↔f32 | **MISSING（AIV）** |
| PACK_NZ P 的 ND→NZ 物理视图 | AIV/VF | 无 NZ 指令面（storage/model.py:27-30 有布局事实；:313-328 甚至拒绝 L1 布局默认） | **MISSING（AIV 布局域）**；TL 有 InsertNd2Nz/ascend_nd2nz_post_copy |
| P_TO_L1 NZ UB→L1 | AIV/MTE3 | 无 | **MISSING（transfer）** |
| PV_OPERANDS（含 hi/lo component） | AIC/MTE1 | 无 | **MISSING（transfer）** |
| hi/lo 拆分 P_hi=cast(P)、P_lo=cast(P-fp32(P_hi))（spec :216-225） | AIV | 无 | **MISSING（AIV 转换域）**；两次 MMA/两 component/额外 staging 须进 effects 计费 |
| UPDATE_U U=a*U+PV | AIV/VF | Vmula 有 | 有语义面，**发射 MISSING** |
| NORMALIZE_STORE U/l+cast+写出 | AIV/VF + MTE3 | Vdiv/UbToGm 有 | **cast + 事件 MISSING** |
| deferred epoch drain（spec :292-313：等旧 anchor 全部 PV→搬 pending→每行乘 exp(a_old-a_new)→clear→发布新 epoch） | MIX 跨 AIC/AIV | **无 state effect、无 epoch 指令、无跨 program barrier**；SS 已登记为不可分配（memory.py:102-107），spec :310 明确 SS 不提供跨 program barrier | **MISSING（completion + state 域）** |

## 3. 任务点名项逐条核查

reduce/sum 缺（Vcadd≠沿 D 归约）；max 缺（无 vmax/vmin）；exp 缺；div 有（无面）；sqrt 有（无面）；bf16↔f32 cast 有 Vcvt 两签名（且 ops.py:352-357 自述 CPU software model、native UNSOURCED），f32→fp16 通用 cast 缺；hi/lo 拆分缺；Cube gemm 全缺；L0C→UB dual copy 缺（仅事实登记）；UB→L1 ND/NZ pack 缺指令面；Aiv FlagSet/FlagWait/Drain **不可发射**（无契约、无面、apply/completion 均不产生）。

## 4. 横向硬缺项：dtype 面（两算子都会立刻撞上）

spec §3 :61 要 AR 四输入 BF16 或 FP16、§4 :174 要 MLA 五输入同 dtype；builder 也接受 ("bfloat16","float16")（attnres/build.py:38、mla/build.py:32）。但 ops.py:51 计算 dtype 只有 f32、:56 载入只有 f32/bf16、emit.py:79 只能写 f32 → **fp16 在指令面与发射面都不存在，bf16 只在 carrier/载入路径存在**；两个 builder 的 dtype 校验比可发射能力宽（现正文只是 f32 64-lane transport 模板，各自已自述为 lowering obligation：attnres/build.py:267、mla/build.py:328）。

## 5. 结论与建议顺序

1. **两算子共同的第一前置是同步事件发射面，不是 Cube。** AR 的 mte2 写→VF 读、mte3 读完成→FREE 都需要事件：PIPE_FACTS（completion/model.py:166 起）声明 mte2/mte3 async、visible_at_issue=False、same_pipe_order=False；且 auto_schedule=False 时 TL 的 InsertSync 整段不跑（ascend/pipeline.py:103-116）。缺 Aiv* 三 op 的契约+面+renderer 时，连 AR 保守 serial 计划也不安全。最小动作：给 FlagSet/FlagWait/Drain 登记 ISA 契约（域=event/predicate）＋ TL_SURFACE 指向已存在的 T.ascend_set_flag/ascend_wait_flag/ascend_pipe_barrier/ascend_threadfence（TLA/sync.py）＋ renderer ＋ completion/plan.py 落成真实语句。**TL 侧无缺项，纯声明/接线，归 MAIN-B。**
2. **然后 AR 可先做 AIV-only**，缺的大多是"声明量"而非"新算力"：给已有 11 条指令补 TL_SURFACE/renderer（vcvt/vmul/vmula/vadds/vmuls/vdiv/vsqrt/vdup/vdupv/vcadd）、给 pset16 补契约；真正需要新注册的是 vmax/vmin 与 vexp（TL 面已存在）。沿 D/N 归约建议登记为 composite VF 单元，不要用 Vcadd 六次链冒充 reduce（spec :112、:245 也要求 composite 显式声明内部合同）。
3. **MLA 必须先补 Cube/MIX，不能先做。** MMA_QK/MMA_PV 是零面且连带 MTE1/FIX/dual_copy/NZ pack 四类 transfer 缺面 + deferred epoch 的 state 效果；还需 tf_target 承认 aic 执行域与 MIX composite 完成合同（spec :245-247）。按 spec :259-263 的分类，这是 **TEMPLATE_MISSING**，不是 TARGET_UNSUPPORTED。
4. **属于"TL 已有、我们只需声明"的缺项**：vexp/vln/vmax/vmin/vdiv/vsqrt/vcvt/vpack/vunpack、reduce_sum/max/min、gemm(MMA)、dual_copy、nd2nz post copy、set/wait flag、pipe barrier、threadfence、get_buf/rls_buf、alloc_l0a/l0b/l0c/l1（实测枚举 TLA/{simd,reduce_op,gemm_op,copy_op,dma,sync,allocate}.py）。AR 的 AIV 缺项几乎全在这类；MLA 的 Cube/MIX 还额外需要 TF 执行域与 composite 合同，不只是声明。

## 6. 每条 MISSING 的最小动作

统一格式：注册 Op（ops.py，ParamDef 效果+pattern）→ typeinfer/verify（_infer_/_verify_）→ schedule_eval（CPU 语义）→ ISA 契约（isa.py：completion/resource/required_region/core_affinity）→ TL_SURFACE + renderer（emit.py）。任一步缺失就是具名拒绝，不用注释代替。
- vmax/vmin、vexp（可选 vln）：五步全做，TL 面现成。
- reduce_sum/reduce_max（沿 D/N）：登记为 composite VF（固定小模板 + 内部顺序声明）。
- f32→bf16/fp16 通用 cast、hi/lo 差分：扩 Vcvt 方向/签名或新增转换条目，并同步 ops.py:51/:56 与 emit.py:79 的 dtype 面。
- Aiv* 事件：契约 + 面 + renderer + completion/plan.py 发射（见 §5.1）。
- pset16：只补 ISA 契约一项。
- Cube MMA / MTE1 / FIX dual copy / NZ pack / P_TO_L1：新增指令族 + aic 执行域承认 + MIX composite 合同（属 MLA 专项，不与 AR 混做）。
- deferred epoch：native state effect + 同组跨 SID 判定（spec :306-310），是 completion 扩展而非指令面。

## 7. UNKNOWN（不作为结论）

(a) vcvt 的真实 PREG 编码与 part 选择：ops.py:352-357 自述 UNSOURCED until 03-20。(b) vcadd 原生归约树形/输出布局：isa.py:129-145 native_evidence_status=UNSOURCED。(c) fp16 是否为本机这些指令族真支持：TL 有 _is_half，硬件面未证。(d) 发射结果能否真编译：本次未 compile。(e) AIC/AIV 配对"一次逻辑 launch"的 MIX 网格细节在 06/07 规格，本轮未核。(f) 上述 implementer 未提交文件的行号可能随其提交漂移。
