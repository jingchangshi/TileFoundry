# Active-only documents — TileFoundry native mainline

- 唯一活动工作树：`TileFoundry_native_mainline`（分支 `feat/ascend950pr-native-mainline-local`，基线 HEAD `ef69b4f`）。
- 唯一 API 规范：`artifacts/native_v3/INTERFACE_FREEZE_NATIVE_MAINLINE.md`；`DECISION_NATIVE_MAINLINE_ONLY.md` 只作策略说明。
- 合同签名：`sha256:a6764633207d175148bcfdbdb6ff2e24eab006d31ac5b8d135f3f90d5752f790`。

## 活动白名单
- `artifacts/native_v3/DECISION_NATIVE_MAINLINE_ONLY.md`
- `artifacts/native_v3/INTERFACE_FREEZE_NATIVE_MAINLINE.md`
- `artifacts/native_v3/CHECKPOINT_MAINLINE_RESET.md`
- `artifacts/native_v3/CURRENT_STATUS.md`
- `artifacts/native_v3/ACTIVE_ONLY.md`
- `artifacts/native_v3/ACTIVE_CONTRACT.md`
- `artifacts/native_v3/research/mainline_cleanup_manifest.md`
- `artifacts/native_v3/research/mainline_import_inventory.md`
- `TileFoundry_native_mainline/ACTIVE_CONTRACT.md`
- `TileFoundry_native_mainline/targets/ascend/ARCHIVE_MANIFEST.md`

## 归档区规则
- `artifacts/native_v3/_archive_compat_defensive/` 只保存历史 checkpoint、旧计划/进度、旧研究/验证/日志与证据，原字节与 sha256 保留。
- 归档区不得作为实现依据、当前验收合同或 PASS 证据；不得引用历史 worktree、历史 task 编号或不完整 PassConfig。
- 原 40 例、原比较器和阈值不变；设备与原 40 数值验收仍为 `NOT_RUN`。

## 禁止项
- 禁止版本分派、兼容 wrapper、unrolled/static materialize、text twin、算子侧 direct `tilelang.compile`、不完整 PassConfig、skip/xfail 正常入口、source-only sidecar、unsupported inventory 假覆盖、防御性 fallback。
- 失败事务必须保留原 entry 并拒绝导出；不允许 metadata-only APPLIED。
