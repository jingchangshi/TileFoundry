# DTYPE-GATE（task-42）— fp16/bf16 贯通性缺项盘点

explorer-g1，只读。基线 HEAD=f341e47；tl_reader.py / codegen/abi.py 等为 implementer 未提交的工作树文件，行号后续需复核。探针：parse 已 emit 源码，未 compile、未 import torch_npu（meta_path 拦截未触发）、未开 /dev。lead 已定的 ①②③④ 层不重复，仅标注出处。

## 1. 新层一：reader / artifact 的 dtype 核对面

结论：**reader 能发现"声明 dtype 与实际 IR dtype 不一致"，但它按构造只能认证 f32**——非 f32 程序既进不来、进来也会被假拒绝。

- 能发现的一侧（实测）：`_compare()` 比 `buffer_map`（codegen/tl_reader.py:506-507 → TL_BUFFER_MAP_MISMATCH），而 buffer_map 三元组含 dtype（expected 侧 :408-415 由 lowered TIR 的 emitted.buffers.dtype 派生；observed 侧 :286/:305 从 parsed PrimFunc 的 buffer 实读）。探针：把 emit 源码 GM 参数整体换成 bfloat16 后 parse → observed buffer_map 报 bfloat16，diff 命中 `['TL_BUFFER_MAP_MISMATCH']`；换 float16 同样命中。**这条核对是真实的、可用 bfloat16/float16 字面量驱动的。**
- 实测更正（本报告最重要的一条）：**observed 侧确实按 IR 实读 dtype，expected 侧却是 f32 字面量** —— observed 走 `str(node.var.dtype)`（tl_reader.py:347→:370-380→:392-393），expected 走 `_vector_dtype(lanes)="float32x%d"`（:211-212，用于 :432/:437/:441）与谓词字面量 `PREDICATE_DTYPE="boolx256"`（:84）。把源码**全部** dtype 站点（GM 参数 + alloc_shared）换成 bfloat16 后 parse：observed 得 `('bfloat16x128','boolx256')`，expected 仍 `('boolx256','float32x64')`，`_compare` 一次命中 **6 个码**：TL_BUFFER_MAP_MISMATCH / TL_ALLOCATION_MISMATCH / TL_BLOCK_MISMATCH / TL_INSTRUCTION_RESULT_DTYPE / TL_INSTRUCTION_OPERANDS / TL_VECTOR_DTYPE_MISMATCH（float16 同样 6 码，`unsupported` 为空 → TL 表面**本身能表达** 16-bit 向量）。→ reader 不是失明，而是**只认证 f32**：任何真 bf16/fp16 kernel 都会得到成批假拒绝。补齐动作不是简单"参数化 dtype"，还要按 dtype 重算 lane/extent——实测同一 64 元素 tile 在 16-bit 表面报成 **x128、access_ptr extent 128**（f32 是 x64/64），说明 lane↔element 映射随位宽变化，而 pset 谓词宽度保持 boolx256。该 x128 语义是否符合 950PR 真机寄存器打包，属需硬件证据项（不可由本探针钉）。
- 无 dtype↔宽度语义规则：`_pset_operands(elem_width)`（:224-225）只把 elem_width 当字面量比；构建事实注释（:70-78）与 ops.py:342-380 的 bf16↔f32 转换粒度 16（ir/execution.py:39-40 SIMD_VF_CONVERSION_WIDTH_BITS）之间**没有核对关系**。实测旁证：16-bit 表面下谓词仍是 boolx256、只有数据侧变 x128 → 谓词粒度与元素位宽的关系目前不由 reader 表达，"pset(32) 配 bf16 carrier"这类不一致只能靠巧合发现。
- ABI / output_count 面：**已经是 dtype-ready**，无需扩。tf_ascend_runtime/abi.py:34-44 RUNTIME_DTYPES 已含 f16→float16、bf16→bfloat16，未知 dtype 具名拒绝 ABI_DTYPE_UNKNOWN（:64-71）；参数 dtype 逐条核对 ABI_PARAMETER_DTYPE（codegen/abi.py:53-55）、绑定核对 ABI_INPUT_DTYPE（abi.py:263-265）；reader 侧还比 ABI 参数序（tl_reader.py:618-625）。→ dtype 缺项**不在 ABI 层**。
- 其余 reader 检查点（供 implementer 参考，与 dtype 无关但同函数）：源码 identity/文件内容一致性（:562-582）、parsed 必须来自本 artifact 源文件（:583-587 → TL_PARSED_PRIMFUNC_NOT_FROM_SOURCE）、Ascend C 单独一层且必须与 TL 源不同并可复现（:591-610）、PASS_CONFIGS 双键（:612-614）。

## 2. 新层二：比较器阈值来源与 dtype 语义（R8 更正后：阈值已实测可得）

- 比较器**不在包内**（work_package/cases/CONTRACT.md:44 原文："原 helper 以 accuracy_mode='cann-bench' 调用 npu_compare。附件未提供 npu_compare 实现，proto 中的 thresholds 为空。……缺失时正式验收 BLOCKED；不能自行猜测 allclose 阈值并称通过原测试。"）。该句约束的是**包内附件不完整**，不表示阈值在任何地方都不可得——见下条 R8 更正。
- 实测佐证：fixtures/{attention_residual,mla}/proto.yaml **存在但无 precision_thresholds 键**；cannbench_contract.py:934 `proto.get('precision_thresholds', {})` → 空 dict，:1008 以 `extra={'precision_thresholds': {}}` 传给 `from npu_compare import compare_case_result`（:982-1006，accuracy_mode='cann-bench'）。cases/case_matrix.json 40 例**零个**容差字段（字段并集只有 attrs/value_ranges/seed/source/execution_status），`status=SPECIFIED_NOT_EXECUTED`、40 例 execution_status 全 NOT_RUN。→ 该空值的影响面比本报告原版所说**小得多**：见第 3 条，proto 的 precision_thresholds 只是 MERE 的可选覆盖，缺省时直接落表内默认阈值。
- **R8 更正（我已逐条自行实测确认，非照抄）**：生效比较器实存于 /home/s00653124/workspace/mla-p16/，npu_compare.py:7 按 accuracy_mode 分派到 cann_bench_compare.py:82-133 的 `CANN_BENCH_THRESHOLDS`，**逐 dtype 具值可查**：float16 threshold=2^-10（small 2^-11、small_error 2^-16、cancel 2^-5）；bfloat16 2^-7（small 2^-8、cancel 2^-3）；float32 2^-13（small 2^-14、small_error 2^-30、cancel 2^-8）；float64 同 f32；hifloat32 2^-11；float8_e4m3 2^-3、float8_e5m2 2^-2；`_FALLBACK_THRESHOLDS = CANN_BENCH_THRESHOLDS["float32"]`（:135）。判定规则：`relative_error = |actual64-golden64|/(|golden64|+1e-7)`，**MERE < threshold 且 MARE < 10·threshold**（docstring :32-34；实现 :823-848），另有 small/cancel 独立通过位（:37-42、:876）。选择函数 `_thresholds_for_dtype`（:1176-1191）：表值为默认，proto 的 precision_thresholds 仅在命中该 dtype 键时覆盖 threshold 一项。
- **f32 代验无法冒充原例的机制证明（本次最关键推论）**：阈值与比较类别一律按 `actual.dtype` 选取（docstring :21-23 明确 `thresholds = table[actual.dtype or float32_fallback]`、`target_golden = cast(golden, actual.dtype)`；实现 :548 `output_dtype = _dtype_name(actual.dtype)`、:787 `golden_target = golden.to(dtype=actual.dtype)`）。→ 若候选输出 f32，该次比较落进 **f32 档（2^-13，比 fp16 2^-10 更严、比 bf16 2^-7 严 64 倍）**且**根本不经过 fp16/bf16 输出档**——它判的是另一个合同对象。叠加 CONTRACT.md:18 要求返回原 dtype，f32 输出连结构合同都不符。**代验不可冒充的结论不变，但依据由"阈值未知"改为"阈值按 actual.dtype 分档、f32 走 f32 档"这一可实证机制。**
- 溯源与禁改约束：mla-p16 属已停用旧 worktree（ACTIVE_CONTRACT.md:25），本节数值是**读到的生效实现事实**，可用于钉门禁与写判据；但**禁止据此修改比较器或阈值**（不得改写阈值、不得另写宽松 allclose 顶替原合同，CONTRACT.md:44 原文仍适用），且 P0 记录仍须登记比较器 module 路径/版本/文件身份（specs/06:37、:17）。
- oracle 侧精度合同（决定代验解释）：CONTRACT.md:46——helper 用 **double** 输入跑 oracle；attention golden 内部强制 **FP32** 中间量；MLA golden 保持 **double**。即"数值参考不要求我们也用 fp16 计算"，但 CONTRACT.md:18 要求候选返回**原 dtype** 的张量 → dtype 是接口合同的一部分。
- dtype 分布实测（case_matrix.json）：**AR bfloat16×17 + float16×3（#3/#9/#17）；MLA float16×15 + bfloat16×5**；supplemental_mla.json 4 例 = fp16×2 + bf16×2。→ 现有 f32-only 面下，**40 例无一例可直接数值验收**；fp16 尤其：ops.py:51/:56、emit.py:79 连 bf16 都只在 carrier/载入路径存在，fp16 在任何位置不存在（lead ①②）。

## 3. f32 代验的可与不可（按 dtype 是否参与算法划分）

判据：dtype 若只是**搬运宽度**（值在 fp16/bf16 与 f32 间可精确或近精确表示，且不改变控制流），可在 f32 上代验**结构/控制流/同步语义**；dtype 若是**算法的一部分**（精度语义、打包、差分补偿），f32 代验无效。合同依据：05_operator_implementation.md:79「数值路径变换是否可用，由原比较器决定」、:214「两次低精度舍入不是原比较器对所有实现的硬要求，若保留应命名为独立 precision policy」。

- **可 f32 代验（限结构层，非数值验收）**：AR 的 D 非对齐/尾块（#1/#5/#7 类，D=2047/1000/5120）、token/N 维度与 R/CN knob 兑现（:151-158 表格要求 IR 实际变化）、V versions/FREE/最后消费者、跨 region guard、STORE_Y 的 MTE3 完成与退出 drain；MLA 的 row mapping/GQA/causal 跳块（:188-197）、列分片 job/grid 计算（:269-290）、layout BSND/BNSD 选择、Dn448 尾片（:265-267）。这些的 dtype 只决定元素宽度，控制流与版本/同步合同与之正交；且 CPU oracle 本身可跑 f16（lead ④：native_reference.py NP_DTYPES 含 f16），说明"能不能算"不是瓶颈。
- **不可 f32 代验（dtype 即算法）**：
  1) **MLA p_hi_lo 两次 MMA 差分补偿**（05:216-225）：`P_lo = cast(P - fp32(P_hi))` 只在低精度目标精度下有意义，f32 路径下 P_lo 恒≈0，hi/lo 语义消失 → 15 个 MLA fp16 例与 p_hi_lo 策略全部不可代验。
  2) **bf16 carrier 打包路径**：Vld dist="UNPK_B16"/_VldDist（ops.py:96-124）、Vcvt 两签名（:342-380）、Vsts pack="PK_B32"（:288-298）本身是 dtype 特定编码，f32 kernel 根本不走这些指令 → AR 的 bf16 载入/回写不可用 f32 代证。
  3) **online softmax / rescale 精度语义**：m/l/U 的 fp32 状态 + exp/max 舍入交互（05:201-214、:292-313 deferred anchor/theta），golden 用 double 而实现用低精度（CONTRACT.md:46）→ 其误差正是比较器要判的对象，f32 代验会系统性低估。
  4) **返回原 dtype 的接口合同**（CONTRACT.md:18）：任何"算完再 cast 成 f32 返回"的代验都不满足原 API 合同，不能算通过该例。
- 结论式：**f32 只能代验"结构与同步是否正确"，不能代验"数值是否过原比较器"**；原 40 例的数值验收必须等 fp16 面闭合（MLA 尤甚：75% 的例是 fp16）。

## 4. 门禁判定式（供 implementer 落地，沿用已定码族，不造同义码）

**"原 40 例可以开始跑" = 下列四项全为真**，任一为假记 BLOCKED（状态词沿用包内既有词表 next_tasks.py:34 与 case_matrix.json 的 SPECIFIED_NOT_EXECUTED，不自造）：

- G1 **指令面**：ops.py 的 VECTOR_DTYPES/LOAD_DTYPES 与 operand pattern 覆盖该例 dtype（fp16 需新增位置；bf16 需计算面而不止 carrier/载入）。判定函数建议 `supported_domain()`（emit.py:802-807）在换 dtype 的 IR 上返回 True；否则其结果码 TL_UNSUPPORTED_DTYPE（emit.py:253-256）即 BLOCKED 依据。
- G2 **发射+reader 一致面**：emit.py:79 TL_DTYPES 放宽**且** expected 侧 dtype/lane 规则化（§1 第 2 条：`_vector_dtype` :211-212 与 `PREDICATE_DTYPE` :84 去 f32 字面量化，并按 dtype 重算 lane/extent——实测 16-bit 表面同一 tile 报 x128/extent 128，非 x64），使 `_compare()` 在 bf16/fp16 真 kernel 上返回 []（消除现必发的 6 码假拒绝，也不误放）。判定式：`read_actual_tl(artifact, parsed, manifest).verified`（tl_reader.py:546，:206-207）为 True。
- G3 **比较器身份已登记（R8 更正：阈值已可得，此项从"解锁阻塞"降级为"合规记录"）**：npu_compare.py:7 → cann_bench_compare.py:82-133 的逐 dtype 阈值已实测确认存在，因此 G3 **不再阻塞原 40 例开跑**；仍必须做的是按 CONTRACT.md:44 / specs/06:37 / specs/06:17 把生效比较器的 module 路径、版本与文件身份（含 CANN_BENCH_THRESHOLDS 的 sha256）写入 P0 记录。缺该登记 → `BLOCKED_DEPENDENCY`（run_contract.py:425 既有码）；已登记但**改动阈值或另写宽松 allclose** → 按违规处理，不是放行条件。
- G4 **该例不属于 §3 不可代验类**：对 fp16/bf16 例，数值验收必须 dtype 原生；f32 结果只能记作结构代验，不得写 PASS。

**判定归属**：G1 由 `supported_domain/emit` 结果码；G2 由 `read_actual_tl` 的 `verified` + TL_VECTOR_DTYPE_MISMATCH/TL_BUFFER_MAP_MISMATCH 明细；G3 由 harness 预检（`preflight_status`，run_contract.py:372-393）；G4 由 case_matrix 的 dtype 与本报告 §3 分类表。四者写进同一 P0 记录；**禁止**修改比较器/阈值或用自写 allclose 替代原合同（CONTRACT.md:44 明文禁止）——阈值虽已可得，可得的只是**读取与登记**，不是改写权。

## 5. 三层缺项计数与最小补齐动作（AR / MLA 分列）

| 类别 | AR | MLA | 最小动作 |
|---|---|---|---|
| 仅缺声明（TL 已有面） | 2：emit.py:79 TL_DTYPES 加 bfloat16/float16；vcvt/vsts 的 pack 面（TLA simd 有 vpack/vunpack/vcvt） | 同左 2 项 + 1：p_hi_lo 的两段 MMA 低精度 operand（TL gemm 支持低精度 P operand） | 各 1 处表项 + 对应 renderer/operand 校验，不动 IR 结构 |
| 需 TF 新注册 / 新签名 | 3：① operand pattern 加 fp16（ops.py:51/:56/:74 全线）；② Vcvt 需 f16↔f32 方向签名（现仅 bf16 carrier，ops.py:342-380）；③ reader expected 侧 dtype+lane 规则化（tl_reader.py:211-212/:84/:432/:437/:441，实测缺一即 6 码假拒绝） | 5：AR 三项 + ④ hi/lo 差分的 cast/子指令面；⑤ Cube gemm 低精度 operand 的 TF 侧注册（0 Cube 面，见 task-38 §2） | 每步都要注册→typeinfer/verify→schedule_eval→ISA 契约→TL_SURFACE→renderer 全链，缺一即具名拒绝 |
| 需真实硬件证据才能钉 | 2：fp16 是否为本机这些 AIV 指令族真支持（本机仅有 REFERENCE 声明，ops.py:352-357 自述 native UNSOURCED）；bf16 carrier 的 PREG 编码/part 选择（同处） | 3：上述两项 + p_hi_lo 的两次低精度舍入能否过原比较器（05:214 明示由比较器决定） | 归 validator 独占设备窗口（HANDOFF_PLAN.md:15），当前一律 REFERENCE_IMPLEMENTATION，不得写 DEVICE_VALIDATED |

**排序建议（与 task-38 结论一致，不重复展开）**：先补 AR 的 bf16/fp16 "声明+reader 一致面"（成本低、解锁 17 个 AR bf16 例的结构代验与潜在数值验收），再谈 fp16 硬件证据；MLA 的 15 个 fp16 例在 Cube 面与 p_hi_lo 硬件证据双双缺位前，**数值上应记 BLOCKED_DEPENDENCY，不应通过 f32 代验冒充**。

## 6. 实测记录（可复跑）

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
  /home/s00653124/workspace/envs/tfnpu-v2/bin/python /tmp/tf_fac/probe_dtype_reader.py
# 基线 _compare(exp,obs) == []。
# 仅换 GM 参数 dtype → 只有 ['TL_BUFFER_MAP_MISMATCH']；全部站点(含 alloc_shared)换 →
#   bf16: observed vector_dtypes=('bfloat16x128','boolx256')，6 码齐发（见 §1 第 2 条），fp16 同理 x128。
# 探针脚本 /tmp/tf_fac/probe_dtype_all.py（parse-only，未 compile、未 import torch_npu）
PYTHONDONTWRITEBYTECODE=1 /home/s00653124/workspace/envs/tfnpu-v2/bin/python -c "import json,collections; \
  c=json.load(open('work_package/cases/case_matrix.json'))['cases']; \
  print(collections.Counter((x['case'],tuple(sorted(set(x['dtypes'])))) for x in c))"
# AR: bfloat16 17 / float16 3 ; MLA: float16 15 / bfloat16 5 ; 容差字段数 0
```

UNKNOWN（不下结论）：(a) 生效比较器的**部署版本身份**（阈值取自停用 worktree mla-p16 的实文件，S4/S5 真正调用的是否同一份仍需 P0 登记确认，specs/06:17）；(b) fp16 在本机 AIV 指令族的真支持面（ops.py:352-357 native UNSOURCED）；(c) p_hi_lo 的两次低精度舍入误差能否落在 fp16 档 2^-10 / MARE<10×thr 内（05:214 明示由比较器决定，且 hifloat32 2^-11 档说明该路径确有独立档位）。三者均需 P0 登记或 validator 设备窗口，非本次可钉。
## 7. 更正记录（R8，2026-11-13 03:2x +08，未静默改）

- **被更正的原结论**：本版初稿 §2 曾写"比较器阈值**并非不可得**"的反面——"附件未提供 npu_compare 实现 → 阈值不可得，必须走正式依赖请求后才能钉门禁"，并在 §1/G3 中把它当作原 40 例的"真正卡点"。
- **更正依据（Lead R8 + 本 explorer 独立复核，逐条实测为真）**：npu_compare.py:7 分派 → cann_bench_compare.py:82-133 `CANN_BENCH_THRESHOLDS`（fp16 2^-10、bf16 2^-7、f32/f64 2^-13、hifloat32 2^-11、fp8e4m3 2^-3、fp8e5m2 2^-2，含 small/cancel 子档）；判定 MERE<thr 且 MARE<10·thr（:32-34、:823-848）；阈值按 `actual.dtype` 选、golden 被 cast 到 actual.dtype（:21-23、:548、:787）；`precision_thresholds` 仅为可选 MERE 覆盖（:1176-1191），故 proto.yaml 缺该键只影响整数/bool 精确语义分支（:712-741、:756-770）。
- **改动范围**：§2 标题与第 1/2/3/4 条、§4 G3、§6 UNKNOWN (a) 项。**未改动**：§1 reader 全节（独立实测，不受 R8 影响）、§3 f32 代验的可/不可分类、§5 三层计数与排序建议、G1/G2/G4。
- **不变的结论**：f32 代验**仍不得冒充原 40 例**，且现在有了更硬的依据——不是"阈值未知所以不能判"，而是"阈值按 actual.dtype 分档：f32 走 2^-13 档、不经过 fp16 2^-10 / bf16 2^-7 档，且原合同要求返回原 dtype"。G3 从"解锁阻塞"降级为"合规登记"，原 40 例开跑的实际阻塞点回到 **G1/G2（指令面 + reader 一致面）**。
- **保留的禁止项**：比较器与阈值**只读**——不得修改阈值、不得另写宽松 allclose 顶替原合同（CONTRACT.md:44）；P0 仍须登记比较器 module 路径/版本/文件身份（specs/06:17、:37）。

