# ACTIVE CONTRACT — TileFoundry Ascend 950PR native mainline

> 本文件在 `artifacts/native_v3/ACTIVE_CONTRACT.md` 与 `TileFoundry_native_mainline/ACTIVE_CONTRACT.md` 两处镜像，字节一致，签名一致。

- 唯一活动工作树：`/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/TileFoundry_native_mainline`（`feat/ascend950pr-native-mainline-local`）。
- 唯一 API 规范：`artifacts/native_v3/INTERFACE_FREEZE_NATIVE_MAINLINE.md`；`DECISION_NATIVE_MAINLINE_ONLY.md` 不再复制 API 签名。
- 合同签名：`sha256:a6764633207d175148bcfdbdb6ff2e24eab006d31ac5b8d135f3f90d5752f790`。
- 当前清理基线：`1e73cb2`（CLEAN-CODE 已完成；新增文件不得回写已归档旧路径）。
- 当前唯一状态：`artifacts/native_v3/CURRENT_STATUS.md`；CLEAN-ARCH 清单：`artifacts/native_v3/research/mainline_cleanup_manifest.md`；当前 import/TL 事实：`artifacts/native_v3/research/mainline_import_inventory.md`。

## 单一主链
Native Scheduled HIR → native UnitTree/effects → 完整 NormalizedProblem → 共享 solver → 唯一 ResolvedPlan → symbolic completion/storage → private candidate module → 显式 TileLang PrimFunc → `read_actual_tl` conformance → `install_entry`/公开导出 → `bind_invoke_once`。

## 强制合同
- ResolvedPlan schema 唯一写者为 MAIN-A/implementer-g2；Lead 只裁定与集成，不并行修改 schema。completion/storage/entry/emit/runtime 消费同一 ResolvedPlan。
- `solve_region(problem, deadline)` 的 deadline 是共享进程 monotonic 绝对截止时间（float 秒）；`None` 仅表示显式无截止；任何阶段不得重置、延长或重新开始预算。
- `build_symbolic_completion(plan, target, effects)` 必须消费 effects；`apply_symbolic_pipeline(...)` 返回 private candidate module 与 source_map；runtime 入口固定为 `bind_invoke_once`。
- 私有 candidate 顺序固定为：apply → private candidate module → `emit_supported_tl`/`read_actual_tl` → 完整 VERIFIED → `install_entry`。`read_actual_tl` conformance 必须在任何 install/export/bind 前通过；失败保持原 entry 且不发布。
- 固定 PassConfig：`tl.enable_auto_schedule=false` 与 `tl.disable_shared_memory_reuse=true`；完整双键配置是当前合同。
- AR/MLA 必须作为 native Module/For/ExecutionScope builder 进入主链；禁止算子侧 direct `tilelang.compile`、自建 HIR 或 text twin 作为实现权威。
- 未知容量保持 UNKNOWN 并拒绝，unknown task/effect/access 必须完整进入 problem 或具名拒绝；失败事务保留原 entry。
- 原 40 例、原比较器和阈值不变；设备与原 40 数值验收仍为 `NOT_RUN`，不得把 CPU/模型结果称 DEVICE_VALIDATED 或阶段 PASS。

## 禁止作为依据
- `artifacts/native_v3/_archive_compat_defensive/` 与所有 `HISTORICAL_WORKTREE_DO_NOT_USE.md` 标记的历史 worktree 只作溯源，不得作为实现或验收依据。
- 禁止版本分派、兼容 wrapper、fallback、unrolled/static materialize、text twin、不完整 PassConfig、skip/xfail 正常入口、source-only sidecar、unsupported inventory 假覆盖。
