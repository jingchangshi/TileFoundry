# Handoff plan — next session

## Entry
Start in /home/s00653124/workspace/TileFoundry_950PR_handoff (branch main, tag handoff-20261112). Do not use the old TileFoundry repository/worktrees.

## Immediate technical state
- Cleanup gate is fail-closed: fake metadata, foreign identity, tampered PassConfig, corrupt TL source, missing parsed PrimFunc/ABI and unverified install are refused.
- The gate currently ends in NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED; real solver/apply/TL reader/export/install conformance is not implemented.
- The old compatibility/static materialize/text twin paths are removed; do not recreate them.
- Device, original 40, P3 dynamic AIV/MIX, P4-P7, cold/hit and independent reproduction remain NOT_RUN.

## Workstreams
1. MAIN-A (implementer): native HIR → UnitTree → complete problem → shared solver → ResolvedPlan → symbolic whole-task apply → actual entry installation; single schema, no compatibility branches.
2. MAIN-B (implementer): SymbolicCompletionPlan → StoragePlan → explicit TileLang emitter → real PrimFunc reader/refinement → bind/invoke → worker isolation; AR/MLA native builders.
3. Validator: independent CPU baseline and candidate checks on detached worktrees; only this role runs device measurements, with /home/.tf_device0.lock and one job.
4. Reviewer: cleanup/candidate review; require actual entry/artifact/ABI/PassConfig evidence.
5. Explorer: original-40 fixture/comparator identity, AR/MLA 20+20, search/cold-hit/P7 preparation.
6. Architect: contract/gate matrix and cross-layer consistency.
7. Challenger: fake-pass and fault mutation on actual artifacts, not documents.

## Gates
- P0 cleanup/contract: current handoff gate; real conformance still missing.
- P1-P3: CPU single-AIV normal entry, then dynamic AIV/MIX with one logical launch.
- P4-P7: AR 20 + MLA 20 original contract, performance families, search/dispatch, cold/hit and independent reproduction.

## Known blockers to close first
- Implement real TL parser/conformance and install verification; do not delete the fail-closed gate when adding it.
- Ensure output_count, ABI, target, parameter types and actual TL source/PassConfig are in entry identity.
- Keep stage and buffer-version domains independent; explicit Fixed/Auto overrides the default Auto upper bound.
- Rebuild active tests after old test archive; zero tests is not PASS.

## Verification command
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests_native_mainline

No NPU execution or original-40 acceptance was performed in this handoff.