# 主链策略切换（2026-11-12 19:45 +08）

## 决策
- TileFoundry Ascend 950PR 后端只实现当前单一原生架构；不提供版本分派、adapter、wrapper、fallback 或历史测试对齐分支。
- 正常入口只允许一条链：Native Scheduled HIR → native UnitTree/effects → 完整 NormalizedProblem → 共享 solver → 唯一 ResolvedPlan → symbolic completion/storage → private candidate module → 显式 TileLang PrimFunc → `read_actual_tl` conformance → `install_entry`/公开导出 → `bind_invoke_once`。
- 禁止用 metadata sidecar、诊断 inventory 或 unsupported 清单替代 solver-facing 问题。未知 task/effect/access 要么以完整 typed problem 进入求解，要么在 source coverage 阶段具名拒绝编译。
- 禁止把 max_stage 与 max_buffer_versions 耦合；Fixed(V) 只能来自作者 BufferSchedule，Auto(min,max) 由共享 solver 在真实约束下决定 V。
- ResolvedPlan schema 的唯一写者是 MAIN-A/implementer-g2；Lead 只裁定与集成，不并行修改 schema。completion、storage、entry、emit 与 runtime 全部消费同一 ResolvedPlan 版本字段。
- `solve_region(problem, deadline)` 的 deadline 是共享进程 monotonic 绝对截止时间（float 秒）；`None` 仅表示显式无截止。任何阶段传递不得重置、延长或重新开始预算。
- 私有 candidate 顺序固定为：`apply_symbolic_pipeline` → private candidate module → `emit_supported_tl` / `read_actual_tl` → 完整 VERIFIED → `install_entry`。`read_actual_tl` conformance 必须在任何 install/export/bind 前通过；失败保持原 entry 且不发布。
- AR/MLA 是两个验收算子的 native builder，返回 native Module/For/ExecutionScope；禁止算子侧 direct `tilelang.compile`、自建 HIR 或 text twin 作为实现权威。
- 仍然保留：原 40 例、原比较器和阈值、真实依赖/同步/容量检查、失败事务不安装、显式 AutoSchedule 关闭；这些是正确性合同，不是兼容层。

## 冻结的职责接口
- 唯一 API 规范以 `artifacts/native_v3/INTERFACE_FREEZE_NATIVE_MAINLINE.md` 为准；本文不复制或定义第二套签名。
- 合同签名：`sha256:a6764633207d175148bcfdbdb6ff2e24eab006d31ac5b8d135f3f90d5752f790`（与 API 规范及两份 ACTIVE_CONTRACT 一致）。
- completion 必须显式消费 effects 参数；apply 必须返回 source_map；runtime 入口固定为 `bind_invoke_once`。
- `read_actual_tl(artifact, parsed_prim_func, refinement_manifest)` 的 conformance verdict、双键 PassConfig identity 和 actual entry/plan identity 必须在本地 commit/导出前通过；失败不安装、不导出。

## 当前写者边界
- MAIN-A/implementer-g2：ResolvedPlan schema 唯一写者；native HIR/UnitTree/symbolic apply/p3_chain 正常入口。
- Lead：裁定与集成、主入口/PassManager 注册；不并行修改 ResolvedPlan schema。
- MAIN-B/flex-1-g2：completion/storage/TL/runtime/AR/MLA native builders 后半链。
- CLEAN-CODE 单一写者：清理生产路径中的当前版本兼容、防御与历史测试对齐残留。
- 其他角色只读复审/验证；不写生产文件。
