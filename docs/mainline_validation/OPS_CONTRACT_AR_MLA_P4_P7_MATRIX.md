# OPS-CONTRACT — AR/MLA 验收与 P4–P7 并行准备（只读）

- 任务：task-116（owner flex-4-g2，explorer）。日期：2026-11-12。
- 性质：**只读准备**。本文件只做事实汇总与并行工作矩阵，供 implementer 在主链（MAIN-A/B/C）完成后直接执行 P4–P7。
- 约束兑现：未编辑任何生产文件；未触碰设备（无 NPU 导入/探测/计时）；未改动任何阈值、比较器、fixture、golden。
- 写范围：仅 `artifacts/native_v3/mainline_validation/`（与本目录 explorer-g2 的 task-112 preflight 产物不冲突：本文件不重跑 preflight，只引用其结果）。

---

## 1. 原 40 身份与比较器合同（P4 验收的权威依据）

### 1.1 fixture 身份

权威输入是原附件包的 `golden.py`、配套测试、实际 cases 与输入生成规则（MASTER_SPEC §目标）。两处包均已核对为**同字节**：

| 文件 | sha256(前16) |
|---|---|
| `fixtures/attention_residual/golden.py` | `cf90b8476c4fabbc` |
| `fixtures/attention_residual/test_attention_residual.py` | `c50fcf953ffe613f` |
| `fixtures/attention_residual/cases.yaml` | `d6d35f240ab979fc` |
| `fixtures/attention_residual/proto.yaml` | `2d85e237072501be` |
| `fixtures/mla/golden.py` | `3996ec1a4bcf142d` |
| `fixtures/mla/test_mla.py` | `f12622752c0d3f6d` |
| `fixtures/mla/cases.yaml` | `57e742e5f1468099` |
| `fixtures/mla/proto.yaml` | `70cd11633ebad052` |
| `fixtures/cannbench_contract.py` | `026f51a94b6a6046` |

- 原始包：`2026111/tf_ascend950pr_single_kernel_v2/fixtures/`（附 `source_manifest.json`，schema `tf-ascend.fixture-sources.v2`，copy_policy=byte_identical）。
- 活动包：`20261112/tf_ascend950pr_native_v3/fixtures/`（task-112 preflight 实际使用的路径）。
- 上表 9 个关键文件两包逐一比对 **全部一致**（2026-11-12 本任务实测）。

### 1.2 比较器链（不可替换）

```
test_<op>.py::main(operator_api)
  └─ cannbench_contract.Harness('level4/<op>', fixture_dir).run(operator_api)
       ├─ proto.yaml['operator'] → precision_thresholds（两个 proto 均无该键 → {}，阈值在比较器内部）
       ├─ golden.py 加载 reference（mla / attention_residual）
       ├─ load_cases(): cases.yaml 为准；断言 case_id 唯一
       ├─ prepare(): seed = sha256(case_id_str)[:8] → int % 2**31（确定性输入）
       ├─ reference_output(): golden 的 '_oracle' 变体，输入先 .double()（fp64 oracle）；'_bench' 变体作 native_ref
       └─ compare(): from npu_compare import compare_case_result(accuracy_mode='cann-bench',
                    extra={'precision_thresholds': ...})，另断言输出为 plain/contiguous Tensor、dtype∈proto
```

- **比较器身份**：`npu_compare.compare_case_result`，accuracy_mode=`cann-bench`。
- **npu_compare 已定位**（task-112 preflight，2026-11-12 20:29，两算子 preflight_status=PASS）：
  `/home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts/npu_compare.py`（sha256 前缀 `ef2506bba`，详见 `preflight_attention_residual.json` / `preflight_mla.json`）。
- **阈值状态**：runner 不推断阈值（"Not inferred by this runner; pin and audit the comparator in P0"）。阈值实体在 npu_compare 内部与 proto 合同中；**P4–P7 任何角色不得查看后"复刻"或替换阈值**，只允许使用安装的比较器原样判定。比较器文件变化 → 判 BLOCKED，不得换用本地阈值。
- **case 标签**：`level4/attention_residual_1..20`、`level4/mla_1..20`（preflight expected_case_ids 已固定，见 task-112 报告）。
- 参考缓存 `CANNBENCH_CACHE_DIR`：run_contract 默认 `off`；P7 cold/hit 时须区分"参考缓存"与"编译缓存"两个身份并分别记录。

### 1.3 原 runner（`tools/run_contract.py`）接线

```
run_contract.py --case {attention_residual|mla} --operator-file <候选.py> [--callable 名] \
                [--device npu:N] [--cache-dir DIR|off] --report <json> [--log <log>] [--preflight-only]
```

- preflight 断言：POSIX 环境、NPU 可见且索引有效、必需文件齐全、`npu_compare.compare_case_result` 可调用、导入的是同一 `cannbench_contract.py`、测试暴露 `main(operator_api)`、**每算子恰好 20 个唯一 case**。
- 正式运行只做数值判定（自述："does not establish schedule conformance or measure performance"）；逐 case 输出 JSON 行 `{"case","passed","outputs"}`，末尾聚合断言 `Failed k/20`；报告 JSON 在成功/失败/阻断三种结局都落盘。
- **operator_api 的来源（P4 关键接线）**：候选必须是主链**实际安装后的入口**（INTERFACE_FREEZE：`install_entry` 真实替换 `Module.functions`，`bind_invoke_once` 单次调用、无 host 数学）。禁止把独立 CPU 模型、internal helper 或文本 twin 当作 operator_file 传入。

## 2. 原 40 case 矩阵（实测自 cases.yaml）

### 2.1 AR（AttentionResidual）20 例

统一签名：`attention_residual(v_stack[N,B,T,D], x[B,T,D], w[D], gamma[D], alpha, epsilon) -> y[B,T,D]`；键路径 RMSNorm→伪查询打分→块轴 softmax→原始 V 加权→门控混合。

| case | N | B | T | D | dtype | alpha | eps | 语义标签（note） |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 1 | 128 | 2048 | bf16 | 0.5 | 1e-6 | default-small 热身 |
| 2 | 2 | 1 | 1 | 4096 | bf16 | 0.5 | 1e-6 | T1 decode / Nmin |
| 3 | 6 | 2 | 256 | 2048 | fp16 | 0.2 | 1e-6 | fp16 small |
| 4 | 3 | 1 | 333 | 2047 | bf16 | 0.5 | 1e-6 | 非 2 幂 D、T 尾 |
| 5 | 8 | 1 | 512 | 1024 | bf16 | 0.95 | 1e-6 | alpha-max 聚合主导 |
| 6 | 8 | 2 | 1024 | 2048 | bf16 | 0.35 | 1e-6 | 中 prefill |
| 7 | 4 | 4 | 512 | 4096 | bf16 | 0.5 | 1e-6 | batch4 |
| 8 | 16 | 1 | 512 | 4096 | bf16 | 0.5 | 1e-6 | deep-stack16 |
| 9 | 4 | 1 | 4096 | 2560 | fp16 | 0.5 | 1e-6 | 长序列 fp16 |
| 10 | 12 | 2 | 512 | 3584 | bf16 | 0.65 | 1e-6 | 中型 |
| 11 | 6 | 1 | 2048 | 4096 | bf16 | 0.5 | 1e-6 | seq2k |
| 12 | 8 | 8 | 256 | 2048 | bf16 | 0.05 | 1e-6 | alpha-min 残差主导 |
| 13 | 4 | 2 | 1024 | 5120 | bf16 | 0.5 | 1e-8 | eps-min |
| 14 | 8 | 1 | 2048 | 1000 | bf16 | 0.5 | 1e-3 | eps-max + 非对齐 D1000 |
| 15 | 24 | 1 | 256 | 4096 | bf16 | 0.5 | 1e-6 | deep-stack24 |
| 16 | 8 | 4 | 2048 | 4096 | bf16 | 0.5 | 1e-6 | 元素量上限档 |
| 17 | 16 | 1 | 4096 | 4096 | fp16 | 0.5 | 1e-6 | fp16 上限档 |
| 18 | 4 | 8 | 512 | 7168 | bf16 | 0.25 | 1e-6 | D=7168 |
| 19 | 8 | 1 | 8192 | 2560 | bf16 | 0.5 | 1e-6 | 长序列 8k |
| 20 | 32 | 1 | 1024 | 2048 | bf16 | 0.5 | 1e-6 | deep-stack32 |

覆盖面：bf16×17 / fp16×3；alpha∈{0.05,0.2,0.25,0.35,0.5,0.65,0.95}；eps∈{1e-8,1e-6,1e-3}；非 2 幂 D（2047/1000）与 T 尾（333/1）；decode T=1 至 T=8192。

### 2.2 MLA 20 例

统一签名：`mla(q_nope, q_rope, k_nope, k_rope, v, numKVHeads=1, scaleValue=-1.0, inputLayout="BSND", is_causal=False) -> y`；V 与 k_nope 数值相同（独立入参，kernel 不得假设共享存储）；scaleValue≤0 时自动 1/sqrt(d_nope+d_rope)；causal 为右下对齐。

| case | layout | dtype | Sq | Nq | Skv | d_nope | d_rope | causal |
|---|---|---|---|---|---|---|---|---|
| 1 | BSND | fp16 | 128 | 128 | 128 | 512 | 64 | 是 |
| 2 | BSND | bf16 | 256 | 128 | 256 | 512 | 64 | 是 |
| 3 | BSND | bf16 | 128 | 64 | 128 | 448 | 64 | 是 |
| 4 | BSND | fp16 | 256 | 128 | 128 | 512 | 64 | 否 |
| 5 | BSND | fp16 | 256 | 128 | 256 | 512 | 64 | 是 |
| 6 | BSND | fp16 | 256 | 128 | 256 | 448 | 64 | 否 |
| 7 | BSND | fp16 | 1 | 128 | 2048 | 512 | 64 | 否 |
| 8 | BSND | fp16 | 1(B16) | 128 | 2048 | 512 | 64 | 否 |
| 9 | BSND | bf16 | 1(B32) | 128 | 2048 | 512 | 64 | 否 |
| 10 | BSND | fp16 | 1(B60) | 128 | 2048 | 512 | 64 | 否 |
| 11 | BSND | fp16 | 1(B128) | 128 | 1024 | 512 | 64 | 否 |
| 12 | BSND | fp16 | 1(B60) | 64 | 2048 | 512 | 64 | 否 |
| 13 | BNSD | fp16 | 128 | 128 | 128 | 512 | 64 | 是 |
| 14 | BNSD | fp16 | 128 | 128 | 128 | 448 | 64 | 是 |
| 15 | BNSD | bf16 | 64 | 32 | 512 | 512 | 64 | 是 |
| 16 | BNSD | fp16 | 1(B60) | 128 | 2048 | 512 | 64 | 否 |
| 17 | BSND | fp16 | 2(B16) | 128 | 2048 | 512 | 64 | 是 |
| 18 | BSND | bf16 | 2(B16) | 32 | 512 | 512 | 64 | 是 |
| 19 | BNSD | fp16 | 2(B16) | 128 | 2048 | 512 | 64 | 否 |
| 20 | BSND | fp16 | 256 | 128 | 256 | 512 | 64 | 是 |

覆盖面：BSND×15 / BNSD×5；fp16×15 / bf16×5；causal 10/10；**Nkv 全部 =1**（官方集已知覆盖缺口）；d_nope 512×17 / 448×3（448 是尾片关键）；decode Sq=1（6 例）、Sq=2 交叉、prefill、Skv 128–2048。case15/18 note 为 bounded-resource 档。

## 3. 补充 case 矩阵（诊断用，不并入原 40）

依据：T21/T24 实施项 + 历史 G63 扩展记录（仅溯源设计参考，非本轮证据）。原则：**补充例只出独立诊断报告，不与原 40 合并计数，不改变官方 20 例集合**。

| # | 补充族 | 触发维度 | 建议最小集 | 归属任务 |
|---|---|---|---|---|
| S1 | GQA Nkv>1 | numKVHeads∈{2,4}, G=32 | ≥3（prefill causal / BNSD / decode） | T21 诊断；T24 分片对照 |
| S2 | explicit scale | scaleValue>0（非自动） | 1–2 | T21 |
| S3 | KV/因果尾 | Skv 非 128 倍数（如 64）、causal 尾块 | 2 | T21（域拒绝 + 数值双记录） |
| S4 | layout 交叉对照 | 同形状 BSND↔BNSD | 1 对 | T21 |
| S5 | D448 尾片 | d_nope=448 分片 vs 不分片，末 64 列覆盖 | 1 对 | T24 |
| S6 | AR D 尾 / alpha 端点 | D∈{2047,1000}、alpha∈{0.05,0.95}（已在原 40 内，作复验锚点） | 复用原例 | T22/T23 |
| S7 | AR 全无效行/NaN 域 | 全 -inf 行、eps 端点 | 2（负例+正例） | T22（对齐 G52 设计） |

执行顺序合同（T21 原文）：补充 shape **先跑 unsharded canonical** 出诊断；sharded 对照留给 T24，不阻塞 P4。

## 4. P4：原 runner 接线 + 单逻辑 launch（读准备清单）

实现者执行序列（每算子同构）：

1. **接线**：主链产出候选模块（真实 install_entry 后的 module）；operator_api = 实际 `module.entry_function()` 等价公开入口；`run_contract.py --case <op> --operator-file <候选> --device npu:0 --report artifacts/native_v3/mainline_validation/p4_<op>_run1.json`。
2. **先 preflight-only**（已完成，PASS，task-112 产物）；正式跑前重跑一次确认环境未漂移（比较 runner 报告内 sha256）。
3. **单逻辑 launch 三项独立验收**（V3-G4 required check "单调用/资源/对应独立通过"，逐项出 verdict，不许合并 PASS）：
   - **单调用**：emitted kernel 恰一个 launch scope；HIR 一个 launch 且包含 spec 全部 pipes；第二次 launch/kernel 具名拒绝。
   - **资源**：live buffers/读流量逐族记录（online/blocked_online/resident_n 三族必须可区分）；StoragePlan 容量来自当前 Target facts；UNKNOWN_CAPACITY 拒绝；资源数字标注 lower_bound_only。
   - **对应（correspondence）**：source/plan/TIR/TL identity 全链一致；`read_actual_tl` conformance 过（parsed PrimFunc 事实 + refinement manifest + 双 PassConfig `tl.enable_auto_schedule=false` / `tl.disable_shared_memory_reuse=true`）；conformance 不过不安装、不导出、不进 runner。
4. **负例随行**：wrong operand、丢 task/依赖、错 slot/generation、缺 writer/tail wait、陈旧 layout offset、陈旧 identity 全部拒绝（INTERFACE_FREEZE 验证边界清单）。
5. AR 先做 AIV canonical；MLA 与 MIX transfer/owner/layout 合同对齐后再 fused（PHASES 关键路径）。MLA 不物化完整 GM scores/P。

## 5. P5：device 计时与性能族（读准备清单）

### 5.1 两套计时协议，身份分开，不得互充

| 协议 | 参数 | 用途 |
|---|---|---|
| 原 bench helper（`fixtures/*/bench_*.py` + Harness.bench_cases） | warmup=3, repeats=5/case | P7 最终 40 性能表 |
| 搜索 harness 预算（spec 08 §8.3，P0 config） | max_candidates_per_case=40（seed≤12/neighbor≤20/advanced≤8），top_k=3，solver 10s，compile 120s，case 搜索 1800s，warmup=10，50 次/批 ×3 确认批，tie_noise=0.02 | P6 搜索 |

### 5.2 计时方法合同（V3-T28）

- 校准过的 device timer 注入；**异步 launch 禁止 wall clock 计时**（DeviceTimerUnavailable 具名拒绝）。
- 计时范围 = 候选 kernel 调用本身；输入准备与参考计算排除；warmup 剔除且丢弃次数记录；统计量（median/p10/p90/spread/相对 spread）必须能从保留的原始样本重算；compile 时间每次单独上报；"一次调用"与"多次 benchmark 迭代"语义分开。
- 每条结果带 `measurement_source`：device 主张必须 DEVICE_TIMER 实采；SIMULATED_CLOCK 只作历史；虚构/模拟/stale 结果具名拒绝。
- **设备互斥**：`flock -n /home/.tf_device0.lock` + 已批准执行窗口；同一时间只允许一个团队测量作业。锁成功≠设备空闲（G69 曾因设备被占 BLOCKED_ENV——先 npu-smi 确认再排程）。疑似挂死立即停该候选，不自行 reset NPU。

### 5.3 P5 各任务读准备要点

- T22（AR 三族）：online / blocked_online / resident_n 真实驻留；blocked_reread 另列；factored 完整 RMS 修复；每族经原比较器；resident 合法域来自真实 StoragePlan；"漏 RMS"反例必须拒绝。
- T23（AR 流水）：R/CN/dchunk/V/stage/VF 每个 knob 改变**实际 IR**；同 source 不同未兑现 knob 不计候选；至少一个正确 pipeline 设备候选；尾 D/N 不越界。
- T24（MLA 列片）：列片后重算 jobs/grid；D448 尾片覆盖末 64 列；BQ/BC 合法 template 探索（未实现≠硬件禁用）；single/hi-lo P 与 FP32 denominator；输出不重叠。
- T25（deferred 状态机）：L0C pending/UB prefix/anchor 与 m/l 关系；旧 epoch 全列全 component PV 完成后才 drain/rescale/clear；迟到 PV/早覆盖 factor 负例由 native 验证拒绝；theta 依整个 epoch 数值域。
- T26（deferred 设备对照）：完整有条件 drain；统一 AIC/AIV predicate；短 KV/阈值跨越/列片/hi-lo 边界覆盖；未实现不得以 PARTIAL 交门。

## 6. P6：搜索代表/竞争/邻域 + dispatch（读准备清单）

- **代表选择**：对计算出的覆盖标签做 greedy set cover（标签：dtype、layout、causal、D/n family wall、KV 长度、workload 档、alpha 端点、GQA、非对齐 D），case_id 决胜保证确定性；AR 8 + MLA 8 个代表（历史记录 AR 代表={1,9,2,5,12,15,3,4}，仅溯源——本轮须在当前 40 上重算，不得直接沿用）。
- **竞争**：每个原 case ≥3 个可行竞争者；只有"实际不同且正确"的产物计数；去重按实际 IR/策略，不按调度字符串。
- **邻域**：winner 邻域每次只改一个因子并报告因子名；接近噪声（tie_noise_fraction 0.02）重测。
- **剪枝具名**：六个资源空间全部计费（GEMM operand staging 取 max、UB 求和；deferred rescale 计 L0C）；resident_n 硬可行墙；未实现 tile 宽度不得作为搜索空间条目；panel 必须整除 d_nope（≤448 且 448%panel==0）；事件池有限（aiv_ready_ids=32）。
- **worker 形态**：主链已有 `search/process.py` 子进程 worker 合同（exec worker，不 fork NPU runtime；argv 显式传路径；结果 JSON 含 `{failure, samples_us, discarded_warmups, measurement_source, sync_note}`，TF_SEARCH_RUN_ID/STAGE/IDENTITY_KEY 环境回显）。T27 的 `search/generator.py`、`search/filter.py`、harness adapters 尚不存在，为 T27 写范围。
- **缓存身份 10 要素**：operator, target_profile, toolchain, source_digest, plan_digest, dtype, layout, shape_digest, precision_policy, candidate_key；source_digest 从实际文件读取；身份不完整→拒绝而非默认。
- **冻结 dispatch（T30）**：覆盖原 40 的 metadata-only dispatch；每 case 恰好一个合法 artifact；无 host 布局/数学补丁；"最快已测"声明必须带样本，不称全局最优。
- **接真构建**（T27）：候选→native→唯一 plan→真实 artifact；粗估排序 / target 硬拒绝 / template 缺失三者分开报告；旧模拟结果不得参与排名。

## 7. P7：cold/hit + 独立复现（读准备清单）

### 7.1 cold/hit（T31）

1. 冻结最终 dispatch；清理本任务编译缓存或显式禁用后，对**全部最终 specialization 重新编译**并完整运行原 20+20（cold）。
2. 随后完整 **cache-hit** 复验原 40，逐 case 核对 source/plan/ABI/flags/binary 绑定与结果一致。
3. cold 与 hit 两次都必须覆盖**精确 40 keys**（无缺项/重复）；每个最终 specialization 有实际重新编译记录；golden/helper 未变（比对 §1.1 hash）；conformance/resource/launch 分别验收。
4. 不是开发日志拼接、不是自制 20 例替代原 runner。
5. 参考缓存（CANNBENCH_CACHE_DIR）与编译缓存身份分开记录；run_contract 默认 off 的行为在报告中显式声明。

### 7.2 独立复现（T32）

- 由**另一个 agent** 仅按文档复现：真实入口 + 真实设备；结果与实际 commit 一致；无 plan/TL/版本 identity 漂移。
- 适用范围跑完整 CPU 回归与 GPU 目标兼容检查；CPU 计数与 skip 原因逐条明确。
- 复现命令链必须可从文档重建：env 解析（`envs/tfnpu-v2/bin/python3.12`，或 TF_ENV_ROOT 覆盖）→ `run_contract.py --preflight-only` → 逐 case 正式运行 → 聚合报告。新失败：定位修复 → 仅复验受影响范围。

## 8. 并行工作矩阵（P4–P7）

lanes 取自 TASKS.json owner 字段（ar / mla / performance / validation / lead）；团队角色映射按本 preset 容量：implementer×2、validator×1、reviewer/challenger 各 1。

| Wave | V3 任务 | 门 | lane→团队角色 | blocked_by | 写范围（互不重叠） | 并行说明 |
|---|---|---|---|---|---|---|
| W1 | T18 AR canonical builder | G4 | ar→implementer#1 | T03,T05,T06,T13（≈MAIN-A/B 主链 + native builder 接口） | `tilefoundry_ascend/attnres/build.py` + attnres 模型/指令 | 与 T20 并行；纯 CPU 准备即可先行 |
| W1 | T20 MLA canonical builder | G4 | mla→implementer#2 | T03,T05,T06,T17（≈MAIN-A/B） | `tilefoundry_ascend/mla/build.py` + mla math/layout/panel 模板 | 与 T18 并行；与 MIX owner 对齐 transfer/owner/layout 合同 |
| W2 | T19 AR 原 20 验收 | G4 | ar→implementer#1 | T01,T16,T18 | AR 正式 operator 入口 + 原 runner 连接 | 与 T21 并行，**设备窗口串行**（flock+窗口）；CPU 侧 conformance/资源/对应可并行 |
| W2 | T21 MLA 原 20 + 补充诊断 | G4 | mla→implementer#2 | T01,T17,T20 | MLA 正式入口 + 原 runner 连接 + supplemental 诊断 | 同上；补充例先 unsharded |
| W3 | T22 AR 三族 | G5 | ar→implementer#1 | T19 | attnres family templates | T22→T23 串行 |
| W3 | T23 AR 流水/驻留/预取 | G5 | ar→implementer#1 | T19,T22 | attnres candidate binding + 策略/cost | 设备候选验证走共享窗口 |
| W3 | T24 MLA 列片/grid/precision | G5 | mla→implementer#2 | T21 | mla sharding + P 精度/packing 模板 | 与 T25 并行（写范围不同） |
| W3 | T25 deferred 状态/effects | G5 | mla→implementer#2 | T08,T10,T21 | mla/rescale.py + completion/state 模板 | 与 T24 并行；CPU 协议模型与 native effects 互核 |
| W3 | T26 deferred 设备对照 | G5 | mla→implementer#2 | T17,T24,T25 | mla deferred 模板 + 设备诊断 | T24/T25 后 |
| W4 | T27 搜索接真构建 | G6 | performance→implementer（W3 结束后释放的槽或 flex） | T12,T14,T15,T19,T21 | search/generator.py、filter.py、adapters | worker 适配可自 P3 起少量真候选预接线 |
| W4 | T28 numeric/launch/profile worker | G6 | performance | T15,T16,T17,T27 | runtime_worker actions + 实测记录 | 计时合同见 §5 |
| W4 | T29 代表矩阵 + 预算搜索 | G6 | performance | T22,T23,T24,T26,T27,T28 | 实际候选结果 + representatives/budget 配置 | 8×8 + 每 case≥3 竞争；设备窗口串行为主瓶颈 |
| W4 | T30 winner 邻域 + dispatch | G6 | performance | T29 | runtime dispatch + 逐 case 性能表 | 重测近噪声对 |
| W5 | T31 最终 40 + cold/hit | G7 | validation→validator | T01,T30 | 最终验收输出 + cold/hit 记录 | validator 执行；lead 复核 |
| W5 | T32 独立复现 + CPU 回归 | G7 | validation→validator（另一 agent 身份） | T31 | 复现说明 + 集成回归记录 | 复现者不得是 T31 执行者 |
| W5 | T33 自包含交付 | G7 | lead | T32 | specs/v3 + REVIEW_HANDOFF/交付索引 | lead 集成收口 |

设备串行化原则：W2–W4 所有设备步骤共享唯一测量窗口；每步只持锁跑一个候选/一算子，CPU 可准备的工作（IR/knob/conformance/资源/代表重算/预算表）全部并行前置。

## 9. 门依赖与假通过防线

| 门 | 阶段离开条件（PHASES.md） | required_checks（GATES.json） | 不足证据（任一出现即不算过） |
|---|---|---|---|
| V3-G4 | 原 20+20 数值、计划对应、资源、单逻辑 launch 分别通过 | 原AR20与MLA20；单调用/资源/对应独立通过 | metadata status alone；standalone model tests alone；direct TL bypass；historical aggregate passed count |
| V3-G5 | 核心族真实实现 + 正确设备候选；关键 knob 改变实际 IR | AR三族真实实现；MLA列片与deferred真实device候选；knob改变实际IR | 同上 |
| V3-G6 | 覆盖目标达到或有具体可行域证据；最快已测候选可重现 | 8代表×8候选与每case3竞争；device时间与邻域；完整单kernel dispatch | 同上 |
| V3-G7 | 无缺 case/假状态/旁路；交付可按文档再次运行 | 最终原40重跑；cold/hit与CPU回归；独立复现和完整交付 | 同上 |

当前基线事实（引用 CURRENT_STATUS/ACTIVE_CONTRACT，2026-11-12）：设备与原 40 数值验收均 **NOT_RUN**；`_archive_historical`（含 G50/G60/G63 的 VERIFIED、G69 记录）只作溯源设计参考，**不得作为本轮实现或验收依据**，其历史结论不得宣称本轮复跑。

## 10. 与现行 board 的衔接

- MAIN-A（task-98）/ MAIN-B（task-99）进行中：T18/T20 的前置是它们产出的主链与 native builder 接口（INTERFACE_FREEZE §职责 API：normalize_intent→…→emit_supported_tl→read_actual_tl→bind_invoke_once）。
- MAIN-C（task-100）端到端集成 + task-113 CPU-BASELINE（ready）先于 W2 的原 40 数值验收。
- task-112（ORIG-40）已交付两算子 preflight PASS 与 npu_compare 定位；P4 执行前按 §1.3 复跑 preflight 比对 hash 即可，无需重做定位。
- 建议：P4–P7 的 board 任务创建时按 §8 表填 blocked_by 与 write_scopes；设备步骤在任务描述里写明"须持 flock 与已批准窗口"。

## 11. 本任务边界声明

- 本文件为只读准备产物；除本文件外未写入任何路径；未运行任何设备/编译/数值命令（除只读 hash 比对与已有 preflight 结果的读取）。
- 所有阈值、golden、cases、比较器、runner 均未改动；P4–P7 执行者同样受此约束（数值协议 P0 冻结）。
- 事实来源：§1–§7 所列路径；历史证据仅注记为溯源。
