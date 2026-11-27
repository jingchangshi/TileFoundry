---
trigger: always_on
description: TileFoundry Ascend evergreen L0-L5 lower-triangular topology, domain redlines, Linked Worktree AgentGates gates, and 4-block falsifiable receipt contract.
---
### R-0. 本仓北极星目标与五条领域常青红线

- **本仓唯一北极星**：同时极小化算子演化**收敛轮数 $N_{\text{rounds}}$** 与**单轮墙钟耗时 $T_{\text{round}}$**，在不可变 5 锚点计分板与单调累积 `guardrail_cases` 约束下单调提升 Ascend 950PR 实卡配对加速比，并通过 L3 全量泛化（双遍 `Delta|Cache| == 0`）与 L4 CANNBench 官方契约。
- **五条本仓常青红线**：
  1. **单一活跃目标绑定（`dav-3510` / `cann_9_3`）**：硬件拓扑参数、方言原语表面（`TILELANG_DSL_SURFACE` 与 `CCE_CPP20_SURFACE`）、模板、静态验证器与离线同步审计工具仅绑定当前唯一活跃硬件与 CANN 主线，严禁引入跨代际条件分支、降级兼容层或惰性占位配置。
  2. **纯物理比值分发（零算子名分支）**：语义前端 `PyTorchSemanticAnalyzer`、相区划分器 `regime_solver.py`（有界 `2..4` 相区与 `_subatom` 隔离）、`EnvelopeInspector` 与 `CapabilityGate` 仅允许依据 `CasePhysicsRatios`、`TopologyAxes`（`3x6x4=72` 正交格）与 7 刻面 `ConstraintRelaxationMask` 路由，严禁按算子名称写分支。
  3. **成本模型默认零副作用（`cost_model_mode="off"`）**：主线依靠 Stage-B 宿主机 `bisheng` 真实资源摘要与 `AgentGates` 实卡配对微突发测速驱动；成本模型处于 `"off"` 时必须保持严格零过滤、零重排、零延迟开销。
  4. **宿主与特权硬件租约物理隔离**：宿主父进程严禁导出 `TILEFOUNDRY_NPU_*` 或在仓内创建虚拟环境目录；NPU 租约、毒丸子进程超时隔离、多卡容灾与 CANN/TSD/ECC 设备故障判定（`rc=75` / `BLOCKED_ENV`）统一由 `AgentGates` 守护进程承载，`INFRA_OCCUPANCY` 与 `INFRA_DEVICE_FAULT` 绝不扣减 `stagnation_rounds` 或污染 Bandit 后验。
  5. **仓内规范与代码卫生硬限额**：根目录 `AGENTS.md` 严格控制在 `<= 4,800 B`（`CLAUDE.md` 保持 `@AGENTS.md`）；采用平铺强类型 Python（禁 `Any`、`cast`、`# type: ignore`、`TYPE_CHECKING`）；生产 `.py` 文件零 `#` 散文注释、单行 `<= 100` 列、文档字符串 `< 8` 行且禁含 RFC2119 关键字或临时任务编号；仓内追踪文件严禁出现宿主机个人绝对路径。

### R-1. 通用九大公理在本仓的常青实例化契约

| 通用公理                | `TileFoundry Ascend` 常青领域契约                                                                                                                                                                                                                                                                                                                                                                                                       | 守护门禁与运行时不变量                                                                              |
|:----------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------- |
| **U-A1 下三角矩阵**     | 严格锁定 $L_0 < L_1 < L_2 < (L_3 \parallel L_4) < L_5$ 六层目录；LLM 合成飞轮层 $L_3$ 与 NPU 执行平面层 $L_4$ 彻底隔离、互不导入，仅在 $L_5$ 通过不可变快照汇合                                                                                                                                                                                                                                                                         | `scripts/dsm_lower_triangular_lint.py` + `scripts/forward_references_lint.py`                       |
| **U-A2 最小信息量**     | 算子规格以 `cases.yaml + proto.yaml` 为唯一真源；编译产物以统一 `ArtifactStore(tl_cache_dir)` 与唯一 `compute_compile_task_key` 为真源；序列化共享统一混入基类                                                                                                                                                                                                                                                                          | 静态门禁阻断跨层重复辅助函数与动态签名探测                                                          |
| **U-A3 规范回执**       | 单步演化严格沿 `CalibrationSnapshot -> RoundSynthesisBundle -> StageBCompileReceipt -> RankedL2Survivors -> PromotionDecision -> RoundReceipt` 单向流转；环境锁竞争耗尽统一抛结构化 `OccupancyExhaustedError(exit_code=75)`                                                                                                                                                                                                             | 协调器间零私有属性直写；`round_funnel_events.jsonl` 100% 精确记录 `fault_plane`                     |
| **U-A4 残差剪枝**       | 保留代码必须直接缩短 $T_{\text{round}}$ 或压降可回收调度残差 $\Delta t_{\text{recov}} = t_{\text{cand}} - t_{\text{roof}} - t_{\text{dispatch}}$；多算子入口 `marathon.py` 仅作为复用单算子控制环的薄调度器                                                                                                                                                                                                                             | 生产入口可达性扫描零孤岛模块                                                                        |
| **U-A5 契约先行**       | 跨模块并发任务必须先在 $L_1$ `agent/contracts.py` 冻结共享不可变数据类与 `Protocol` 端口签名，再拉出各 Linked Worktree                                                                                                                                                                                                                                                                                                                  | 分支合入前核验 `agent/contracts.py` 契约完整性                                                      |
| **U-A6 语义值域商集**   | **Stage-B 值域编译闭环**：先在内存对当前相区 `cases.yaml ∪ guardrail_cases` 全集运行候选 `selector` 提取 `CompileTaskSpec`（统一涵盖单核、1-to-N 子核与 Track-B 实例化单元），在值域按 `task_key` 求商集后于宿主机 CPU（`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）一次性跑 `bisheng` 落盘 `ArtifactStore` 与 `prewarm_manifest_v1.json`，并严格结合该 `spec` 自身的 `--cce-disable-vf-stack-reserved-ubuf` 标志与 `stack_bytes > 0` 判定栈溢出 | L2/L3 全程 `compiled_tasks == 0`；全集特化容量与栈溢出 100% 在 Stage-B CPU 侧闭环，零漏网、零假阳性 |
| **U-A7 机械棘轮**       | 全量仓库静态检查钩子统一注册于 `.pre-commit-config.yaml`；`scripts/audit_tilelang_surface_sync.py` 持续保障 `TILELANG_DSL_SURFACE` 与上游 `tilelang.ascend.language` 导出原语 100% 同步；方言渲染器按活跃拓扑与症状刻面无截断装配规则                                                                                                                                                                                                   | `.pre-commit-config.yaml` 全钩子退出 `0`                                                            |
| **U-A8 协议测试**       | `tests/ascend/` 按 `l0..l5` 与源码六层同构组织（另设 `benchmarks/` 承载实卡宿主契约），全部通过显式 `Protocol` 注入替身并采用参数化等价类表验证物理守恒律（双侧物理底噪、批次标准误 $\sigma_{\text{paired}}$、同 Case 对齐 $\eta$、解耦后 `13+1` 症状单纯形）                                                                                                                                                                           | `scripts/test_hygiene_lint.py` + `tilefoundry-npu-cpu` 官方门禁全绿                                 |
| **U-A9 双类提交与收据** | 结构重整提交与语义演进提交严格分立；每次提审附带绑定当前 Worktree `source.root` 的 `AgentGates` 收据 ID 与 950PR 5 锚点配对差分表                                                                                                                                                                                                                                                                                                       | `agent-gates verify-requirements` 返回 `satisfied=true`                                             |

### R-2. 本仓六层严格下三角目录拓扑（$L_0 \to L_5$）

- **$L_0$ 硬件事实、SIL 与编译存储层**（`ascend/target/`、`ascend/sil/`、`ascend/passes/`、`ascend/compile_cache/`、`ascend/cost_model/bisheng_res_parser.py`、`ascend/templates/`）：硬件常量、方言表面、`TileLangSemanticGate`（亚毫秒级静态拦截 `@T.prim_func` 内部嵌套函数、受保护导出重绑、非法 `T.Stage` 嵌套及漏写 `enable_offset=True`）、SIL 14 红线、唯一 `ArtifactStore` 与 `compute_compile_task_key`。**零上层依赖。**
- **$L_1$ 算子规格、守恒代数与因果知识库层**（`agent/contracts.py`、`agent/specs/`、`agent/kb/`、`ascend/cost_model/experience_lake.py`）：`cases.yaml + proto.yaml` 注册表、不可变 5 锚点与单调 `guardrail_cases`、受空核 `null_launch_us` 与参考算子实测耗时双侧约束的 `calibrate_dispatch_floor_us`、融合配对 MAD 与批次标准误的 `compute_paired_sigma_us`、严格同 `anchor_case_idx` 对齐的效率 $\eta$、与 $(1 - \eta)$ 乘子解耦的 `13+1` 症状单纯形、以及显式声明 `allowed_dialects` 方言域隔离的因果规则库与贝叶斯追踪器。**仅依赖 $L_0$。**
- **$L_2$ 静态路由与统一 Stage-A/B 编译门禁层**（`agent/router/`、`agent/gates/`、`ascend/schedule/`）：单遍 TIR/AST 水位线检查器（两快照比对 `ObjectivePhysicalLedger`）与唯一 **Post-Selector `StageBCompileGate`**。**仅依赖 $L_0 \dots L_1$。**
- **$L_3$ 候选合成、可达闭包与因果飞轮层**（`agent/orchestrator/`、`agent/flywheel/`）：纯合成角色前置零工具声明、轻量头注释 + 原生 fenced code 协议、`<read_only_host_tail>` 只读宿主尾部隔离、无字节硬上限的可达调用图闭包 + Host-Device Launcher ABI 切片提取、仅针对真语法/API 错误启用的流式活性看门狗修复（`NOOP_IDENTICAL` 与片上容量/栈溢出修复次数恒为 `0`）、后台异步投机 `explore_creative` 轨道，以及 $3\sigma_{\text{paired}}$ 增益触发的消融蒸馏器。**仅依赖 $L_0 \dots L_2$，与 $L_4$ 互不导入。**
- **$L_4$ NPU 执行平面层**（`agent/npu/`、`agent/probes/`）：`AgentGates` Linked Worktree `--source` 透传、`O(1)` 预热清单命中校验、毒丸子进程配对微突发评测、`OccupancyExhaustedError`、L3 双遍幂等门禁、L4 CANNBench 验收与按 Case 键控的 `msprof` 采集。**仅依赖 $L_0 \dots L_2$，与 $L_3$ 互不导入。**
- **$L_5$ 战役控制环与薄 CLI 层**（`agent/campaign/`）：纯不可变 `CampaignStateSnapshot` 推进器、Top-2 L2 优胜者同轮递补晋级 L3 协调器、遥测账本写入器与多算子薄调度外壳。**依赖 $L_0 \dots L_4$。**

### R-3. 本仓物理挂载布局与日常标准命令手册（R-SOP-1 ~ R-SOP-6）

#### 1 & 2. 拉起 Linked Git Worktree 并投喂极简派单指令

```bash
git -C <TILEFOUNDRY_ROOT> worktree add -b <type>/<topic> ../wt_<topic> main
```

因通用宪法与仓内路由已自动挂载，每次向 `agy` 派单仅需给出四要素：

- **目标与物理判据**：一句话说明要压降的耗时项或修复的边界不变量；
- **层属与独占文件白名单**：指明唯一所属层 `L<k>` 及允许触碰的相对路径列表；
- **提交拆分指令**：要求先交 `refactor:` 负熵清理提交（若涉及结构调整），再交 `feat:/fix:` 最小语义与正反例测试提交；
- **门禁与收据要求**：跑通洁净环境 `agent-gates --source ../wt_<topic>` 门禁并输出四联收据。

#### 3. 本地全量静态门禁自检（由 `pre-commit` 自动拦截，亦可显式全跑）

```bash
pre-commit run --all-files && python3 scripts/audit_tilelang_surface_sync.py
```

#### 4. 洁净非 Sourced Shell 下执行 `AgentGates` 官方 CPU 门禁签核

```bash
env -i HOME="$HOME" PATH="/usr/bin:/bin:$PATH" \
  <AGENT_GATES_ROOT>/.venv/bin/agent-gates submit \
  --workspace tilefoundry \
  --source <WORKTREE_ROOT> \
  --gate tilefoundry-npu-cpu

env -i HOME="$HOME" PATH="/usr/bin:/bin:$PATH" \
  <AGENT_GATES_ROOT>/.venv/bin/agent-gates verify-requirements \
  --workspace tilefoundry \
  --source <WORKTREE_ROOT>
```

#### 5. Ascend 950PR 实卡 Linked Worktree 六项常青验收判据

涉及编译闭环、提示词合成、延迟代数或 NPU 执行平面的变更，须在 Linked Worktree 上以 `--device-ids 0,1` 实跑代表性纯 TileLang（Track-A）算子与混合/C++（Track-B）算子各 6 轮，同时满足以下 6 项常青不变量：

1. **编译幂等与零残留**：全轮次 Prewarm `compiled_tasks == 0`，L3 满足双遍 `Delta|Cache| == 0`，子进程退出后零临时目录泄漏。
2. **Stage-B 值域全闭环**：全集 `cases.yaml` 的片上容量与栈溢出 100% 在 Stage-B 宿主机 CPU 侧拦截，零漏入 L2/L3 实卡、零合法分支假阳性误杀。
3. **物理延迟守恒**：`calibrated_dispatch_floor_us` 严受 `null_launch_us` 与参考算子实测上界约束，真实未达屋脊 Case 的 `recoverable_sched_us > 0`。
4. **症状单纯形非盲**：存在正硬件计数器瓶颈时 `13` 个因果症状非零，不发生伪效率坍缩。
5. **基础设施故障免疫**：遭遇共享宿主机占卡或设备异常（`rc=75`）时自动退避/切卡，且 `stagnation_rounds` 与 Bandit 后验保持零污染。
6. **真卡签名晋级收据**：晋级候选取得绑定当前 `<WORKTREE_ROOT>` 的 `AgentGates` L3 `PASS` 收据，5 个计分板锚点满足 $3\sigma_{\text{paired}}$ 非退化守恒。

#### 6. 60 秒审阅四联收据、线性合入与 Worktree 回收

```bash
git -C <TILEFOUNDRY_ROOT> merge --ff-only <type>/<topic>
git -C <TILEFOUNDRY_ROOT> worktree remove ../wt_<topic>
git -C <TILEFOUNDRY_ROOT> worktree prune
git -C <TILEFOUNDRY_ROOT> branch -d <type>/<topic>
```

