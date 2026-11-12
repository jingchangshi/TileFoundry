# HANDOFF 理解（Lead 综合，2026-11-12 21:2x +08）

> 本文是 Lead 对 `docs/HANDOFF_PLAN.md` 交接任务的理解与核验结论。所有结论都经过实际读取/运行；每条给证据位置或命令。
> 写作基线：`git rev-parse HEAD` = **f341e47529453bdbc8dd81f2c108d36628fb4b8b**；交接标签 `handoff-20261112` = **374de776**（tag 指向 3 个提交的清理历史 e94e652→6424bc1→374de77；内容源 commit 964ebb0）。tag 与 HEAD 之间的 2 个提交**不是交接内容**，是并行团队在本会话期间提交的（见 §6）。

## 0. 一页结论

1. **任务本质**：把 TileFoundry 的 Ascend950PR 后端做成**一条**原生主链——Native Scheduled HIR → UnitTree/effects → 完整 NormalizedProblem → 共享 solver → 唯一 ResolvedPlan → 符号 completion/storage → 私有 candidate → 显式 TileLang PrimFunc → `read_actual_tl` conformance → `install_entry`/导出 → `bind_invoke_once`，再用 attention_residual(AR) 与 MLA 两个算子走这条链，最终过原 40 例（AR20+MLA20）、P3 动态 AIV/MIX、P4–P7 性能/搜索/cold-hit/独立复现。
2. **交接交给我们的不是一个能跑的编译器，而是一个"能证明自己拒绝"的骨架**：13 个冻结 API 里 **0 个可执行**——7 个是具名 fail-closed stub（`raise MainlineNotImplemented`），6 个连定义都没有（`normalize_intent`/`build_native_unit_tree`/`analyze_native_problem`/`resolve_schedule`/`read_actual_tl`/`bind_invoke_once`）；5 个核心 schema 类型（RegionBindingSummary/UnitTree/NormalizedProblem/ResolvedPlan/SymbolicCompletionPlan）全仓库零定义；共 12 处 `raise MainlineNotImplemented`——即**没有任何一个冻结职责已完成实现**。
3. **导出 gate 是真的，但它既不完整也没被强制**：`require_export_ready` 的本地检查真实存在，却无条件追加 `NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED`（p3_chain.py:201）所以恒 False；同时 `src/` 里**没有任何消费者**调用它，正常入口只用 `raising=False`（p3_chain.py:242）记一条 metadata。challenger 58 条探针证明：除这一行外，plan/completion/storage/ABI/parsed/conformance 全可用鸭子对象伪造（§4）。
4. **活动测试面很薄**：`targets/ascend/tests_native_mainline` 在 tag 上 2 文件 12 用例，现在 4 文件 **24 passed**（其中 12 个来自并行团队的 ABI/identity 提交）。这些测试**不覆盖正常入口**，也不证明完整的 AR/MLA 算式（数值用例只验证 `y=x*v` / `out=q*k` 这类乘法模板）。
5. **输入有缺口**：`work_package/` 缺 `implementation/`（4 个参考实现文件）、`evidence/`、`plan/`（34 任务 DAG + 8 阶段门），导致 `work_package/tests/` 三个测试在交接树内直接 `ModuleNotFoundError`。原始工作包完整存在于磁盘（`…/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3`），见 §5 D1。
6. **交接不是静止的**：另一个团队正在同一工作树提交（314e016、f341e47）并写 `docs/mainline_validation/HANDOFF_E_*`、`run-…-task136/`。任何"当前状态"结论都必须写明它依据哪个 commit（§6）。
7. **未跑过的东西一件都没变**：NPU 执行、原 40 数值、P3 动态 AIV/MIX、P4–P7、cold/hit、独立复现，在交接文档与本轮证据范围内全部 **NOT_RUN**（其他会话是否跑过：UNKNOWN，本仓库无对应证据）。

## 1. 权威文档层级（读的顺序就是权威顺序）

| 层 | 文档 | 作用 | 备注 |
|---|---|---|---|
| 1 | `docs/HANDOFF_PLAN.md` | 交接方写给下一会话的入口/工作流/已知阻塞 | 35 行，最先读 |
| 2 | `README_HANDOFF.md` | 快照来源、清理范围、任务背景、验收命令 | 与 HANDOFF_PLAN 一致 |
| 3 | `docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md` | **唯一 API 规范**（13 个职责 API + 事务 8 步 + 写者边界 + 验证边界） | 合同签名 `sha256:a6764633…` |
| 4 | `docs/ACTIVE_CONTRACT.md` | 强制合同/禁止项（与 INTERFACE_FREEZE 镜像） | **行为约束有效，但其中"唯一活动工作树"指向旧树，对本树已过时** |
| 5 | `docs/DECISION_NATIVE_MAINLINE_ONLY.md` | 单链策略与理由 | 与 3/4 一致 |
| 6 | 规格：`work_package/MASTER_SPEC.md`(172KB, 8 章+34 任务) 与 `work_package/specs/00–07` | 实施细节：solver 约束式、符号物化算法、completion 协议、TL 发射/refinement、worker、验收 | 是实现的主要依据 |
| 7 | `docs/CURRENT_STATUS.md`、`docs/ACTIVE_ONLY.md`、`docs/research/*`、`docs/mainline_validation/*` | 清理/验证记录 | `CURRENT_STATUS`/`ACTIVE_ONLY`/`research/*` 描述**旧工作树**（模块清单与本树不符）；`mainline_validation/` 混有当前交接树的身份核对（`HANDOFF_E_ORIG_IDENTITY_ACCEPTANCE.md`，task-139）与旧日志，需逐文件判定 |

## 2. 合同里最关键的 12 条（实现时必须逐条兑现）

1. 唯一生产链；禁止版本分派、兼容 wrapper、fallback、unrolled/static materialize、text twin、算子侧 direct `tilelang.compile`、source-only sidecar、假 unsupported 覆盖。
2. `PipelineSpec` 单 schema：`max_stage`（stage 上界）与 `max_buffer_versions`（auto 上界）**域独立**；`Fixed(V)` 只能来自作者 BufferSchedule。
3. `solve_region(problem, deadline)`：deadline 是**共享进程 monotonic 绝对截止时间**（float 秒），`None` 才是显式无界；任何阶段不得重置/延长。
4. ResolvedPlan 是唯一 authority，schema 唯一写者（MAIN-A）；completion/storage/entry/emit/runtime 消费**同一版本字段**。
5. `build_symbolic_completion(plan, target, effects)` 必须显式消费 effects；不按 N 展开（site/channel 数由有限 task/消费者数决定）。
6. StoragePlan：版本确定后再分配；UNKNOWN_CAPACITY **拒绝**，不得用 footprint 回填。
7. 私有 candidate 顺序固定：`apply_symbolic_pipeline` → private candidate → `emit_supported_tl`/`read_actual_tl` → 完整 VERIFIED → `install_entry`。**read_actual_tl 必须先于任何 install/export/bind**；失败保持原 entry 且不发布。
8. 固定双键 PassConfig：`tl.enable_auto_schedule=false` + `tl.disable_shared_memory_reuse=true`；实际编译调用点要能读出这两个键。
9. entry/artifact identity 必须覆盖 output_count、ABI（顺序/shape/dtype/storage/scalar）、target、参数类型、实际 TL source 与 PassConfig。
10. 未知 task/effect/access 要么完整进入 problem，要么具名拒绝；不允许"投影时静默丢弃"。
11. AR/MLA 只能作为 native Module/For/ExecutionScope builder 进链；host 只做元数据校验、输出分配、一次调用，无 host 数学。
12. 原 40 例、原比较器与阈值不变；缺依赖是 BLOCKED，不是 PASS；CPU/模型结果不得称 DEVICE_VALIDATED 或阶段 PASS。

## 3. 当前真实状态（分层）

| 层 | 状态 | 证据 |
|---|---|---|
| 类型/schema（PipelineSpec/TaskIntent/BufferSchedule） | **真实**，单 schema、无版本分支、字段校验完整 | `src/tilefoundry/ir/hir/pipeline.py:96-328` |
| parser/printer/CLI 单 schema | 真实（`tf.pipelined(stop, max_stage=…, max_buffer_versions=…)`） | `src/tilefoundry/parser/pattern_nodes.py:5200-5257` |
| 正常入口 | **可达但未跑通纵向切片**：`cli finalize → checked_program → analyze → finalize(BEFORE_LOWER → ConvertHIRToTIR → AFTER_LOWER)` 存在且被 pass 注册证明可达；我实测最小 Ascend 程序 `analyze` 通过、`finalize` 在 `ConvertHIRToTIR` 报 `acc instruction issue has no lexical mesh`（缺作者级 SimdVF/mesh scope），完整链实跑 **UNKNOWN** | `src/tilefoundry/cli/schedule.py:30-40`、`schedule/finalize.py:12-19`；本会话 /tmp 探针 |
| Ascend target facts | 真实（topology aiv/aic 兄弟域、内存层级、space/event facts、completion_ops） | `targets/ascend/tilefoundry_ascend/tf_target.py:89-217` |
| 类型与事实数据 | 真实：event namespace(`completion/model.py` 564 行)、storage layout/bytes(`storage/model.py` 347)、space/transfer(`memory.py`)、ISA 契约(`ir/isa.py`) | explorer-g1 STATE_MAP §3 |
| 已实现但**孤立**（不在链上） | `storage/placement.py`（196 行完成闭区间放置引擎）、`bf16.py`、`search/*`、`ir/isa.py`、`completion/model.py` | 同上 |
| 13 个冻结 API | **0 可执行**；7 stub + 6 MISSING | STATE_MAP §2 |
| 12 处 `raise MainlineNotImplemented` | 存在（plan 4 / solver 1 / completion 3 / storage 3 / codegen 1） | STATE_MAP §1 |
| 主链 pass | `RunAscendNativeMainline` 第一步 `compile_mainline` 即 `stopped_at="freeze"` | `passes/p3_chain.py:224-226` |
| 导出 gate | 本地检查真实 + **无条件缺口** :201 + **无强制消费者** | `passes/p3_chain.py:171-212,242`；我在 `src/` grep 无 `require_export_ready|EXPORT_GATE_KEY` 引用 |
| D0 七 pass（仍注册在正常链） | 部分真实、部分空转：`ValidatePipelineDeclarations`/`VerifyPipelinePreservation`/`VerifyTirPartSelection` 等仍做真实校验；**谓词覆盖环节**恒空（`requirements` 恒 `()`、`_collect_actual_predicates` 恒 `{}`），因此覆盖性检查永远通过而不能证明覆盖 | `passes/pipeline.py:689-697,745,781-793,862,873` |
| 残留模板形态的 metadata 键 | `PREDICATE_PROVENANCE_KEY = "d0.predicate_provenance"`（占位符字面量确实存在于源码；成因与是否对应真实秘密泄漏 **UNKNOWN**） | `passes/pipeline.py:963`（待修） |
| AR/MLA builder | 真实 native Module/For/ExecutionScope 骨架 + typed metadata；**算式是 stub**（AR 只 `x*v`，参数缺 w/gamma/alpha/epsilon；MLA 只 `q_nope*k_nope`） | `attnres/build.py:175-269`、`mla/build.py:233-330`、`backend_operators/test_operator_abi_numeric.py:66,114` |
| ABI（并行团队新增） | 真实：ordered ABI、trailing output_count、纯运行时包无 TF/torch 依赖 | `tf_ascend_runtime/abi.py`、`codegen/abi.py` |
| identity（并行团队新增） | 真实：按 Op 属性参数 + Var binder 的结构 identity | `plan/identity.py`、`passes/p3_chain.py` |
| 活动测试 | tag: 2 文件 12 passed；HEAD: 4 文件 **24 passed** | 实测命令见 §7 |
| 设备/原 40/P3/P4–P7/cold-hit/独立复现 | **NOT_RUN** | `docs/HANDOFF_PLAN.md:10`、`ACTIVE_CONTRACT.md:22` |

## 4. 交接主张核验（claim → 结论）

| 交接主张 | 核验结论 | 证据 |
|---|---|---|
| "Cleanup gate is fail-closed：fake metadata / foreign identity / 篡改 PassConfig / corrupt TL / 缺 parsed PrimFunc+ABI / 未验证 install 都被拒" | **部分成立**：直接调 `require_export_ready(raising=True)` 时上述都拒；但**没有生产路径强制它**，且除 :201 外本地检查可被鸭子对象整体伪造 | challenger 58 探针（`docs/handoff/CHALLENGE_PROBES.md`）：完整鸭子 ChainRecord + 匹配 identity 假工件 → 缺口只剩 :201；`src/` 无 gate 消费者 |
| "gate 以 NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED 结束" | **成立**：p3_chain.py:201 无条件追加 | 直接读码 |
| "真实 solver/apply/TL reader/export/install conformance 未实现" | **成立且更严重**：主链前半 6 个 API 连定义都没有，5 个关键 schema 类型不存在 | STATE_MAP §2 |
| "旧兼容/static materialize/text twin 路径已移除，不要重建" | **成立**：`attnres.twin`/`mla.kernel`/`completion.materialize`/`codegen.native_tl` 均 ModuleNotFoundError；正常入口 import 不拉 tilelang/torch_npu | challenger 探针 §"版本差异与旁路证据" |
| "设备、原 40、P3、P4–P7、cold/hit、独立复现 NOT_RUN" | **成立** | 同上，且本会话未触设备 |
| "输出 count/ABI/target/参数类型/实际 TL source 必须在 entry identity 里" | **是要求，尚未全部兑现**：identity 已含 output_count/参数类型/Op 属性；ABI 的 ordered digest 未进门比对；target identity 只 class/name；TL/source identity 只查非空或字符串相等 | CONTRACT_GATE_MATRIX C25/E07/E08；p3_chain.py:136 读 `abi.input_count` 而真实 `OrderedABI` **没有该属性**（会误报 `ABI_INPUT_COUNT_MISMATCH`） |
| "stage 与 buffer-version 域独立；Fixed/Auto 覆盖默认上界" | **schema 层成立**（pipeline.py:65-120、passes/pipeline.py:201）；**solver 层不存在**，无从验证 | CONTRACT_GATE_MATRIX C02/C03 |
| "重建 active tests；零测试不是 PASS" | **成立且紧急**：当前 24 个测试没有一条跑正常入口、没有一条覆盖主链/fail-closed 矩阵；pytest 零筛选 exit 5 | challenger 实测 exit 5；STATE_MAP §3 |
| "不要用旧 TileFoundry 仓库/worktree" | **成立**；原始**工作包**（只读输入）不是 worktree，见 §5 D1 | HANDOFF_PLAN:4、README_HANDOFF |

## 5. 输入缺口与 Lead 决定

**缺口（实测，explorer 逐项核对）**：`PACKAGE_MANIFEST.json` 列 82 项，交接树 `work_package/` 只有 56 项（+manifest 自身），**缺 26 项**：`evidence/` 17、`implementation/` 6、`plan/` 3；在盘 56 项 sha256/bytes 全部一致，无清单外文件。原始包 82/82 全部一致。后果：
- `work_package/tests/` 三个文件（`test_modulo_kernel.py`/`test_symbolic_template.py`/`test_compile_transaction.py`）在交接树内直接 `ModuleNotFoundError: implementation`（我实测，唯一直接原因就是缺 `implementation/`）；`work_package/tools/verify_package.py` 在交接树 `FileNotFoundError: plan/TASKS.json`，在原始包则 **PASS**（18 fixture / 40 cases / 34 tasks / 8 gates）。
- 规格里点名要"迁入"的四个参考实现（`modulo_kernel.py` 362 行固定 II Z3 核 + 独立整数 postcheck、`symbolic_template.py` 117 行、`native_bindings.py` 148 行、`compile_transaction.py` 83 行）与 34 任务 DAG/8 阶段门（`plan/TASKS.json` 1095 行、`GATES.json`、`PHASES.md`）都不在交接树里。
- **原始工作包完整存在于** `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3`（无 `.git`，不是 worktree；`tools/verify_package.py` 在它上面 PASS：fixture 18 / cases 40 / tasks 34 / gates 8）。
- 禁令范围核对：HANDOFF_PLAN/README_HANDOFF/ACTIVE_CONTRACT 禁的是「旧 TileFoundry 仓库 / worktree / 被 `HISTORICAL_WORKTREE_DO_NOT_USE.md` 标记的历史 worktree」；而 `docs/research/mainline_cleanup_manifest.md` 执行顺序第 3 条明写「**保留只读工作包 20261112/tf_ascend950pr_native_v3 及 reference 原样**」。即：把原始包当只读输入不禁，禁止的是拿旧仓库/worktree 当实现依据或 workspace。（旧 worktree `TileFoundry_native_mainline` 两个路径**均已不存在**；旧仓库 `/workspace/TileFoundry` 仍在 main@f4d6fe1。）

**D1（建议，待用户确认）**：把原始工作包当**只读参考输入**（不是实现依据、不是 worktree）：允许阅读/移植 `implementation/modulo_kernel.py` 的固定 II 约束核（规格 §3.6 与 START_TEAM 明确要求迁入 `src/tilefoundry/vendor/tilelang_z3` 单一路径）、`symbolic_template.py` 的 p/k/i 构造、`native_bindings.py` 的 binder 克隆思路、`compile_transaction.py` 的事务编排；**禁止**复制其测试壳、禁止把 text twin/静态展开带回来、所有生产代码必须落在冻结合同规定的模块并配新测试。理由：不移植就要从散文重推 Z3 约束核，代价高且容易出现"第二套 solver"。

**D2（建议）**：D0 七个 pass 从正常链移除注册，把仍必要的声明/谓词/part 检查职责迁入 `normalize_intent`/`analyze_native_problem` 与 actual verifier（architect 裁定，CONTRACT_GATE_MATRIX §"D0 逐项"）。注意交接文档里"必须从正常链移除"这句出自 `docs/research/mainline_import_inventory.md` §1.2，不是 cleanup manifest 原文。

## 6. 并行工作流（重要）

本工作区**不是我们独占**。在我这个会话进行期间，另一个团队在同一棵树提交：

| 提交 | 作者 | 内容 |
|---|---|---|
| 314e016 | flex-1-g3 | AR/MLA 输出改为 trailing ABI 参数；新增 `codegen/abi.py`、`tf_ascend_runtime/abi.py`(283 行)、`backend_operators/{native_reference.py,test_operator_abi_numeric.py}` |
| f341e47 | (账号) | identity 改为按真实 Op 属性 + Var binder；新增 `plan/identity.py`(193 行)、`test_identity.py`(162 行) |

另有未跟踪产物：`docs/mainline_validation/HANDOFF_E_ORIG_IDENTITY_ACCEPTANCE.md`（task-139）、`run-20261112T210101+0800-task136/`、以及他们建的 detached worktree `TileFoundry_task136_cpu_matrix_…`。

**协调规则（本会话团队遵守）**：只读调查只写 `docs/handoff/`；不写 `src/`、`targets/`、`tests/`；不 `git add/commit/checkout/worktree`；报告必须注明依据的 commit。**任何生产实现开始前，必须先与并行团队确认写者归属**，否则必然撞车（尤其 `passes/p3_chain.py`、`plan/*`、`codegen/*`）。

## 7. 环境与工具链事实（实测）

- Python `3.12.16`（`/home/s00653124/workspace/envs/tfnpu-v2/bin/python`）；`z3 4.15.4`、`tilelang 0.1.15+cpu.git0e687c17`（本地 wheel，CPU 构建）、`torch 2.12.0+cpu`、`ortools 9.15.6755`、`numpy`、`pytest` 均可导入。
- **`torch_npu` 裸导入失败**（缺 `libhccl.so`），必须先 `source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh`。这直接约束 runtime worker 与所有设备阶段入口（specs/04 §7）。
- 设备：`npu-smi info` 显示 3 张 **Ascend950PR**，Health 全 OK；三卡均有他人 python 进程驻留（勿干扰）。设备锁 `/home/.tf_device0.lock` 存在，实测 mode `-rw-rw-rw-`、`test -w` 通过、`flock -n` 可获取（validator 报告写"普通用户不可写"，以本次直接实测为准）。按合同：只有 validator 角色做设备测量，串行 + 单 job。
- TL 版本注意：规格把参考 TL 钉在 `558d53b491925364a9b95a26cb5af66a7e7462fe`，实际部署是 `0e687c17`（不同版本）。P0 必须先做 capability probe 并记录差异（specs/04 §3.1/§6.5）。
- 原 40 输入身份：18 份 fixture 与 `work_package/fixtures/source_manifest.json` 逐字节一致；40 = AR20 + MLA20；比较器 `npu_compare.compare_case_result(accuracy_mode='cann-bench')`，阈值在比较器内部（proto 无 threshold 键），不得替换。
- 验收命令（当前可跑）：
```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
/home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests_native_mainline
```
  - tag 374de77：`12 passed`；HEAD f341e47：`24 passed`（构成实测：`test_operator_abi_numeric` 7 + `test_identity` 5 + `test_native_builders` 8 + `test_search_rank` 4；validator 报告的 4/9 是笔误，以 `--collect-only` 实测为准）。
  - 仓库通用 `tests/` **不能当回归基线**：实测 `759 passed / 231 failed / 5 skipped / 67 collection errors`（exit 1）。分组：`tests/core` 725P/2F/1E、`tests/nvidia` 21P/201F/5S/62E、`tests/tilefoundry` 13P/25F/1E、`tests/apple` 0P/3F/3E。根因：PYTHONPATH 缺 `targets/nvidia|apple`（59+3 个整文件导入失败）、`safetensors`/`transformers` 未装（4）、旧 API 消失（`PIPELINE_INTENT_SCHEMA`，1）、`PipelineSpec.num_stages` 漂移（25）。这 67 个收集错误与主链无关；主链自身零测试。`README_HANDOFF.md:48` 已写明这类历史材料是**有意排除**的。`tests/core` 的 725 通过只能当弱参考。

## 8. P0 阻塞项（按依赖排序）与建议任务

| # | 事项 | 交付物 / 验收 | 依赖 |
|---|---|---|---|
| P0-1 | 把 fail-closed 拒绝矩阵落成正式 active 测试（18–24 条），并修 3 个缺陷：`CHAIN_KEY` 非 ChainRecord → 命名拒绝；`abi.input_count` 与真实 OrderedABI 不一致；`PREDICATE_PROVENANCE_KEY` 占位符 | `targets/ascend/tests_native_mainline/test_export_gate.py`；`pytest …tests_native_mainline` 全绿且新测试含正向"真实产物"分支 | challenger 报告 §推荐 |
| P0-2 | 给导出 gate 一个**真实消费者**（`compile`/build 边界 `raising=True`），拒绝 cached False | 新增拒绝码 `EXPORT_GATE_NOT_ENFORCED` + 测试 | P0-1 |
| P0-3 | 真实 solver：移植固定 II Z3 核到 `vendor/tilelang_z3` 单一路径，实现 `solve_region(problem, deadline)` + 独立整数 postcheck + source postcheck | solver 单测（小域穷举对照、fixed-II、fixed-V、owner 资源分域、环形跨 II、容量、UNKNOWN≠UNSAT） | D1 决定 |
| P0-4 | ResolvedPlan/RegionPlan 唯一 schema（`plan/`），含 region Expr 引用、II/origin/max_shift/tasks/buffers | schema 测试 + 篡改拒绝 | P0-3 |
| P0-5 | NormalizedProblem/UnitTree（`normalize_intent`/`build_native_unit_tree`/`analyze_native_problem`），whole-VF、effects、未知具名拒绝 | 覆盖/反例测试；百万 N 常量结构 | P0-3 前可并行 |
| P0-6 | P1 作者样例两份（动态 AIV pipeline、有限 MIX）真正跑通 `checked_program→analyze→finalize` 到主链 | 正常入口纵向切片测试 | P0-5 |
| P0-7 | completion/storage/apply/verify/install/TL emit/read_actual_tl 后半链 | 私有 candidate→VERIFIED→install 顺序测试 | P0-4 |
| P0-8 | ~~与并行团队划定写者边界~~ **已由用户停止并行会话解决**；本树独占 | — | 已完成 |

## 9. 用户裁定（2026-11-12 21:4x）

1. **D1 = 批准**：原始工作包 `implementation/` 可作只读参考移植——固定 II 约束核迁入 `src/tilefoundry/vendor/tilelang_z3` 的**单一**求解路径（TF/TL adapter 共用同一 core）；`symbolic_template.py` 的 p/k/i 构造与 `native_bindings.py` 的 binder 身份克隆/入口替换思路用于 plan 与 apply；`compile_transaction.py` 只作编排参考，不新增第二套驱动。
2. **D2 = 批准**：七个 D0 pass 从正常链移除注册；仍必要的职责（PipelineSpec 规范化、tail/part 声明、TIR 后的真实谓词覆盖核对）迁入分析阶段（`normalize_intent` + `analyze_native_problem` + actual verifier）。这是 S0 之前的结构性一步。
3. **协调 = 已解决**：用户已停止并行会话；本工作树由本会话独占（HEAD 仍 `f341e47`）。并行会话留下的 detached worktree `TileFoundry_task136_cpu_matrix_…`（374de77）与未跟踪的 `docs/mainline_validation/HANDOFF_E_*` 保留原样，不清除。

## 10. 证据索引（本目录）

- `STATE_MAP.md`（explorer-g1）：13 API 判定、模块分类、tag→HEAD 差异、结构性问题。
- `CHALLENGE_PROBES.md` + `probes/`（challenger）：58 条可复跑探针、鸭子接缝表、加固清单、移除条件。
- `CONTRACT_GATE_MATRIX.md`（architect）：C01–C33 合同矩阵、E01–E12 移除条件、MAIN-A/B 切分与单写者范围、D0 裁定。
- `VALIDATION_BASELINE.md`（validator）：mainline 24/24、legacy tests/ 分组计数与根因、环境/设备只读事实。
- `INPUT_FACTS.md`（explorer）：82 项清单缺 26 项的逐项核对、fixture 20 文件身份、残留 import 扫描（333 .py → 330 OK）、禁令范围原文。
- `REVIEW_UNDERSTANDING.md`（reviewer）：对本文的 10 条更正建议；已并入本文（tag 历史、算式结论、D0 空转范围、正常入口实跑、占位符成因、测试构成等）。
