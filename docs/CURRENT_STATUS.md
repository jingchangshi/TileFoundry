# CURRENT STATUS — native mainline (2026-11-12 20:40 +08)

- 唯一活动工作树：`TileFoundry_native_mainline`，分支 `feat/ascend950pr-native-mainline-local`；清理基线 `1e73cb2`，当前文档 HEAD `df692e3`。
- 唯一 API 规范：`artifacts/native_v3/INTERFACE_FREEZE_NATIVE_MAINLINE.md`；当前状态只以本文件为准。
- 合同签名：`sha256:a6764633207d175148bcfdbdb6ff2e24eab006d31ac5b8d135f3f90d5752f790`。

## 清理与合同状态
- CLEAN-ARCH（task-107）：completed，只读清单已落盘 `research/mainline_cleanup_manifest.md`。
- CLEAN-DOCS（task-108）：completed，活动白名单/合同已建立，214 个历史文件已原字节归档并生成 SHA-256 索引与 ARCHIVE_MANIFEST。
- CLEAN-CODE（task-109）：completed（`55dc3ce` 归档移动 + `1e73cb2` 编译器清理/文档）。
- CLEAN-CODE-2（task-117）：in_progress，并行 native builders；只允许新增 `attnres/build.py`、`mla/build.py`、`search/rank.py` 与 `tests_native_mainline` 新测试，不作为 CLEAN-VERIFY 的清理门。
- CLEAN-RESIDUAL（task-125）：in_progress；`df692e3` 初审为 NEEDS_CHANGES，正在修 stage/V 独立（max_stage 不得当 buffer 版本上界）、`require_export_ready` 伪 READY（必须读 actual entry/plan identity/TL conformance）、移除 `annotations.enable_offset` 旧入口、归档 M6-SCOPE-NOTES。
- CLEAN-VERIFY（task-110）：in_progress，blocked by task-108/task-109/task-125；不再依赖 task-117。
- MAIN-G（task-104）：completed，当前输出 `research/mainline_import_inventory.md` 已保留为活动资料。
- CONTRACT-FIX（task-118）：completed（文档）；ACTIVE_ONLY 签名已同步，两份 ACTIVE_CONTRACT 镜像一致。
- STATUS-SYNC（task-130）：completed（仅本文件）；修正 task-110 依赖并记录 `df692e3`/task-125 缺口。

## 待验证边界
- ResolvedPlan 唯一写者、deadline 语义、private candidate/read_actual_tl/install 顺序尚待 task-110 清理后复核。
- 设备与原 40 数值验收仍为 `NOT_RUN`；无阶段 PASS，不继承归档区旧 CPU PASS。
- 原 fixture、比较器与阈值未改动。
- 归档区不是实现依据；历史 worktree 已加标记，不删除、不 reset、不 clean、不改分支历史。
