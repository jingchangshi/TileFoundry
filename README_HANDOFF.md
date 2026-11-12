# TileFoundry Ascend 950PR — clean handoff

## Source
- Source snapshot: commit 964ebb034f34ecbda312b2ef319320f7fdec26e9; the old source worktree was removed after handoff.
- This directory is a standalone snapshot with a new git history; it has no old TileFoundry branch or worktree history.
- Generated from the commit above; old compatibility/defensive/historical archive directories are excluded.

## Cleanup performed
- Removed old TileFoundry worktrees including native_v3, native_mainline, main_review, p3_02, task71/80/85/92 validation, task113 baseline and /tmp worktrees.
- Deleted old feature branches: feat/ascend950pr-native-mainline, feat/ascend950pr-native-mainline-local, feat/ascend950pr-native-v3, feat/p3-02-resolve-schedule, feat/tf-ascend950pr.
- The original TileFoundry repository now retains only its main branch; do not use it as the new workspace.
- New work should start in this handoff directory (branch main, tag handoff-20261112).

## Current status
- Cleanup state: fail-closed export/install gate verified by independent reviewer and challenger probes.
- The cleanup PASS only covers rejection of fake metadata/foreign identity/PassConfig tampering/unverified install.
- Real solver/apply/TL reader/export/install conformance is NOT_IMPLEMENTED; do not treat the clean gate as a working compiler chain.
- Device, original 40, P3 dynamic AIV/MIX, P4-P7, cold/hit and independent reproduction remain NOT_RUN.

## Active documents
- docs/ACTIVE_ONLY.md
- docs/ACTIVE_CONTRACT.md
- docs/CURRENT_STATUS.md
- docs/DECISION_NATIVE_MAINLINE_ONLY.md
- docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md
- docs/CHECKPOINT_MAINLINE_RESET.md
- docs/research/mainline_cleanup_manifest.md
- docs/research/mainline_import_inventory.md
- docs/mainline_validation/ (current preflight/identity reports only)

## Task context at handoff
- Implementer-g2: task-125 CLEAN-RESIDUAL commit 964ebb0; next A-READY/MAIN-A.
- Flex-1-g2: task-117/132 committed; next B-READY/MAIN-B.
- Architect: task-110 CLEAN-VERIFY not signed; independent reviewer/challenger probes passed the fail-closed gate.
- Reviewer/validator/challenger/explorer tasks remain on the shared board; no phase PASS is claimed.
- Original contract fixtures and comparator thresholds are preserved; no threshold or comparator change.

## Verification commands
```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH=src:targets/ascend \
/home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests_native_mainline
```
- No NPU execution in this handoff.
- No original-40 numeric acceptance in this handoff.

## Inputs
- work_package/ contains the read-only source specs, fixtures, reference core, cases and tools; historical evidence/implementation/plan are intentionally excluded.
