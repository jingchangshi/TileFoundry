# Mainline reset checkpoint (2026-11-12 19:55 +08)

- Worktree: /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/TileFoundry_native_mainline
- Branch: feat/ascend950pr-native-mainline-local, HEAD ef69b4f, clean at creation
- Strategy: single TileFoundry Ascend 950PR native mainline; no v1/legacy compatibility, no defensive shims/sidecars, no old-test backstop. Records: DECISION_NATIVE_MAINLINE_ONLY.md, INTERFACE_FREEZE_NATIVE_MAINLINE.md.
- Baseline CPU: targeted 8 native files 116 passed / 1 failed; failure is an old test asserting single-key PassConfig and legacy flag strings, to be rewritten/removed, not preserved.
- Fresh task DAG: task-98 MAIN-A (compiler chain), task-99 MAIN-B (TL/runtime/operators), task-100 MAIN-C (integration), task-101 MAIN-D (validator), task-102 MAIN-E (reviewer), task-103 MAIN-F (challenger), task-104 MAIN-G (explorer), task-105 MAIN-H (architect review).
- Fresh roles running: implementer-g2, flex-1-g2, reviewer-g2, flex-2-g2, flex-3-g2, explorer-g2; architect spawned for task-105.
- No device kernel, no original 40 numeric run, no phase PASS. Goal remains active.
