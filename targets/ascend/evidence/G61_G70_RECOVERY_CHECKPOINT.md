# G61/G69/G70 recovery checkpoint — 2026-11-12

Status: INCOMPLETE. No NPU kernel was executed in this recovery round. G61/G70 and final G71/G72 are not PASS. The user requests continued implementation, real device evidence, strict search budgets, dispatch and local commits. User authorizes the existing worktree and all six handed-off dirty files.

## Source ownership and commits

- Worktree: `../TileFoundry_p3_02`; branch `feat/p3-02-resolve-schedule`.
- Starting HEAD: `878a363271cdc79c0df3e8cbc413ef2a037cadd8`.
- `d31f359`: strict aggregate candidate budget, deadline validation, stop admitting candidates after case deadline, bounded argv worker process helper and five regressions.
- Implementation HEAD at this checkpoint: `01c990f`: always terminate the owned subprocess group after timeout even when its leader exits on TERM; real child-ignores-TERM regression.
- Sole writer and NPU executor: implementer-g1. Lead/reviewer/validator/explorer are read-only. Never reset/clean/stash/overwrite user changes. Local commits only.

All original dirty files are preserved: `evidence/G69_SEARCH_HARNESS.json`, `tests/test_search_g69.py`, `tilefoundry_ascend/search/filter.py`, untracked `evidence/G61_DEVICE_EVIDENCE.json`, `tests/test_mla_g61.py`, `tilefoundry_ascend/mla/rescale.py` (paths relative to targets/ascend). These are not silently included in the two fixes above.

## Actual verification in this round

From integration worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend ../envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests/test_search_deadlines.py targets/ascend/tests/test_search_g69.py
```

Result: exit 0, **50 passed in 1.04s** after descendant cleanup fix. Earlier run before that fix: 49 passed in 0.53s. These are CPU search-mechanism tests, not device measurements. `git diff --check` passed before commits.

`search/process.py` provides a real subprocess timeout with retained native stdout/stderr logs and a monotonic shared Deadline. Its process group belongs only to the newly started worker. **It is not yet connected to a G70 driver.** Existing injectable `run_candidate()` still calls compile in-process and checks elapsed time afterwards. Do not describe that path as enforceably bounded. Existing `search_case` only stops starting new candidates once its wall deadline expires; the eventual device driver must pass remaining shared budgets to worker execution.

Lead reports package `verify_package.py` failed because the reference tree contains extra pytest caches and `evidence/G43_TOOLCHAIN_BLOCKER.json`. This round did not rerun that check or modify reference/manifest. Recheck using a temporary copy containing precisely original manifest files; retain the original failure record.

## Environment and commands

- Host environment, no new container. Interpreter `../envs/tfnpu-v2/bin/python` resolves to the existing tilefoundry-npu-dev Python 3.12 environment.
- `source /data/pri/9.3.0-weekly/cann-9.3.0/set_env.sh` makes Torch and TileLang imports work. Torch reports `2.12.0+cpu` with torch_npu backend extension; this string alone does not prove NPU execution. Without CANN source, import fails with missing `libhccl.so`.
- `npu-smi info`: Ascend950PR healthy OK, driver utility 25.7.rc1. Device 0 had other-user/root pytest/python processes. Never reset, sudo or signal those processes.
- Every future device invocation must hold `flock -w 60 /home/.tf_device0.lock`; no device invocation occurred yet. Avoid measuring amid unrelated active jobs; do independent CPU work meanwhile.
- Prior device experimentation is under sibling `../mla-tl-run/`; it is not the production repository. `g61_deferred.py` there explicitly runs protocol arithmetic only.

## Independent reviewer/validator findings that block acceptance

1. **G61 real backend absent.** `mla/rescale.py` is a Python scale-epoch/protocol model; it does not emit native TF IR or a deferred MixedKernel. BoundaryObligation strings are not real instruction effects/completion edges. Need anchor, pending L0C, UB prefix, final flush; published common group predicate/factor retained across epochs; all columns and hi/lo complete before drain/rescale/clear/new-PV. Model tests cannot replace device evidence. Existing JSON must remain partial. The older evidence script also reports mismatched-guards refused=false because it inverted a legal check, although a unit mismatch test exists.
2. **Formal production compilation bypass.** `mla/kernel.py` directly calls `tilelang.compile` and uses fixed `T.Pipelined`/versions with automatic scheduling. It consumes neither TF native HIR nor unique ResolvedPlan/TIR/PassManager output. `codegen/x_runtime.py` compile/bind/invoke/sync paths remain UNIMPLEMENTED; lowering can return no source/module.
3. **AR production callable absent.** `attnres/kernel.py` is ABI/config, `kernel_cn.py` a summary, `hir.py` a separate descriptive dataclass graph; `twin.py` emits source strings and substring checks. Previously executed AR implementations reside in sibling experimental scripts. Need real production `attention_residual` wrapper and native TF chain, not merely import an experiment and attach a sidecar plan.
4. **Conformance checker unsound for final gate.** `codegen/twin.py` reads slot from instance comments instead of actual slot assignment; guard comparison is missing; sync compares role/channel without full generation/slot/paired endpoints. AR checker has no authoritative ResolvedPlan/TIR inputs and does not enforce its parsed autoschedule value. A native plan mutation must affect actual compiled IR/cache identity. Mutating actual slot/guard/wait endpoint must be rejected; consistent event renumbering and legal address relocation may pass.
5. **Sharding occupancy bug.** MLA wrapper chooses cores from unsharded jobs; builder multiplies jobs by shards but min(cores,jobs) retains old grid. B1/S1/Nq128/Nkv1/BM32 shard1/2/4 has jobs4/8/16 yet grid4/4/4. Fix before real comparison. D448 has seven 64-wide panels: equal shards are 1 or 7, not 2/4. Need legal original device evidence.
6. **Tail/fullmask gaps.** Existing kernel rejects KV tails not multiple of 128 and small head groups rather than padding. G63 tail examples are all multiples of 128. Final normalize lacks the l=0 guard claimed by G60 JSON. Must execute short loop, true KV tail, tail query rows, allmask and GQA/explicit scale tests.
7. **Original numerical provenance incomplete.** Old `g60_all20.py` replays oracle cache by case string and lacks oracle/input/golden hash binding; handwritten harness does not exit nonzero on failed case. Reviewer verified external golden/test/cases/proto hashes currently match authoritative fixtures and observed old 20PASS logs, but this does not satisfy final launch/conformance/performance evidence. `g63_ext.py` uses custom thresholds. Final dispatch must run authoritative `tools/run_contract.py` / original `test.main`. Helper forces cann-bench mode with precision_thresholds (original proto has no overrides, so comparator defaults apply); npu-contract default is not this run mode.
8. **G69/G70 incomplete.** G69 JSON VERIFIED conflicts with its simulated timing/device BLOCKED status. No G70 driver/dispatch exists. Freeze actual upper/time/device/oracle/global budgets; enumerate only real executable differences; never count different labels/seed/flag IDs as candidates. G70 requires each operator >=8 representatives with >=8 distinct correct measured candidates, each original case >=3 competitors, winner neighborhood expansion, required AR online/blocked/resident and MLA canonical/deferred/shard routes. Budget exhaustion cannot waive these.

## Next authorized steps

1. Lead review deadline commits; independent validator reruns tests. Keep G69 labelled partial until integration.
2. Implement a minimal **real native TF production chain**, starting AR as the simpler operator: native author IR -> finalize/type/semantic checks -> one authoritative ResolvedPlan -> completed/stored TIR -> explicit executable twin preserving task/loop/guard/slot/core/completion mapping, with TL automatic rescheduling disabled. Do not make a parallel compiler or no-op/identity evaluator. Existing fake/text twins are not sufficient infrastructure.
3. Move necessary known-working device implementation into production while making its actual schedule/data path derive from native artifacts; add mutation-sensitive verifier regressions and compiler-artifact checks. Then extend MLA and implement real deferred route.
4. Fix shard grid, actual tail/fullmask safety and epoch effects. Use the original comparator and representative device cases under flock, record raw artifacts/trace/identity.
5. Connect strict subprocess deadlines to real candidate driver, generate meaningful candidates, original-check and measure on device with raw samples/interleaved confirmations, checkpoint reasons/unmeasured routes. Build dispatch only from verified measured winners.
6. Final G71 authoritative 20+20 through dispatch plus independent launch/plan/storage/performance evidence; G72 reproduction bundle and local commits. No push.

## Resumed native production micro-case checkpoint (2026-11-12 12:25 +08)

This section supersedes the earlier implementation status only for the restricted AR RMS arithmetic primitive. Full AR/MLA/G61/G70 remain incomplete.

- `3d71167`: fixed real static instance induction/tuple/guard substitution and normalized one-trip wrapper; nested For/unknown statement rejected only in instruction materialize_v2. Tests use tracked pipe_vadd_carrier, not an untracked author file.
- `f5e8ace`: output Copy resource matching now normalizes aiv.mte3 vs mte3; Copy read enters the unique plan, V_MTE3 READY and MTE3_V FREE are derived. Materialized SMEM Let bindings now use the same physical views as actual operands. Explicit original_tir input to scheduled verifier checks expected binding addresses, including missing/all-wrong Let cases. Scope remains static instance names, not general lexical binder or symbolic pipeline support.
- Native bridge commit following f5e8ace: native_square.py is two-input x*z (z=x gives AR squared-input primitive), four 64-element f32 tiles. native_aiv.py traverses final actual TIR; x_runtime.compile_tilelang calls real TL with tl.enable_auto_schedule=False; unique ResolvedPlan.emit_for_codegen permits only typed native binding, legacy paths still refuse. This is no full AR implementation.
- Actual generated source and CCE are retained in native_aiv_run_20261112. TL SHA256=022e44a121c84bcc864f99b7483f56508471506662420bf488f8210109f1a5e9; CCE SHA256=97169f6eef0a98bd426c8d6455406dc2cc065a00874ede2149f15f37b1ab8fe5 (6872 bytes). CCE contains four SIMD vmul, real GM offsets and V_MTE3/MTE3_V events; no dummy arithmetic. Exact CCE bytes preserved, including generated trailing blank line (git diff --check flags that generated formatting only).
- Compile job bash-4c89b117b7bc49b8 completed exit0 after explicit-plan compilation, no set_device or launch. Source/CCE/identity summary: native_aiv_run_20261112/result.json; logs are honestly named compile_extract.txt and device_init_extract.txt, transcribed extracts rather than raw complete logs. Session history export was unavailable.
- Device job bash-42292cd687dc4d78 completed exit1 at torch.npu.set_device(0), BEFORE loading author/compiling/allocating/launching. Error507033 and E39006 Failed_Send_OPP_To_Device / TsdOpen failure. No device numerical PASS. No launch retry, reset, sudo or signals to other users.
- Both background jobs were collected; no remaining running jobs.
- CPU: Lead independently ran native+materialization+G33+G42 = **52 passed in8.59s**. Implementer native fault tests =8 passed in2.13s; after output regression addition materialization=4 passed in1.29s. Do not equate these with device correctness.
- Compile-only exact reproducible CLI is now native_aiv_device.py --compile-only; see native_aiv_run_20261112/REPRODUCE.md. This CLI was added after the successful inline compile and has not been rerun this round. Device mode tests x!=z and square, exact equality, exits nonzero on failure.
- Own existing container read-only inspection shows /dev/davinci0,/dev/davinci1,/dev/davinci_manager,/dev/hisi_hdc mappings and shared /home,/data; no container execution or NPU initialization was attempted. Host npu-smi remains healthy but other-user jobs remain resident.

Next authorized round: read-only diagnose existing device environment/init failure; resume locked micro-case product/square numerical verification when valid; then fix remaining native limited-domain constraints and extend full AR/MLA. Do not restart device launch before Lead continuation. Original six handed-off dirty files remain preserved and uncommitted.

## Device environment handoff after resumed round (2026-11-12)

User is helping repair environment. No further initialization/kernel probes are to start before feedback. Head native refinement f61b2f6; 13 native CPU tests pass. Host507033/E39006 now traced to unreadable root-only CANN aicpu_hccl.tar.gz. Existing own root container reads it and TSD opens successfully, but the single authorized micro-case run exited139 without kernel output. Exact Python crash phase unknown; report FAIL_ENV/unknown-stage, not a kernel hang. New phase-aware runner and imports-only mode committed; imports-only passed, no set_device.

Evidence: native_aiv_device_20261112/HANDOFF.md, original host debug plog, original copied own container plog, raw stdout/stderr and import logs. Jobs bash-71c09b42b7a94098(exit139), bash-c0758e6dac854a47(exit0) collected; no running job/owned container worker. Future user command must use container-inner timeout with host flock. No reset/sudo/other-process signal or package permission change. Lead has exact nonexecuted init and product/square commands.

Write ownership changed: second implementer owns search/harness.py,search/process.py and new CPU worker-deadline tests; this implementer must not edit/stage those files. Original six dirty + user review directory remain preserved. Full fingerprint/Counter actual Evaluate coverage requested by reviewer is deferred to next round; current census is only type/order/count plus existing grouped operand map validation.

### Reviewer follow-up: unresolved arithmetic operand correspondence

After f61b2f6, actual noncompletion operation coverage checks only operation kinds/order/count. A real counterexample remains: after full finalize, change the first Vmul Evaluate args from (va,vb,mask,dst) to (vb,vb,mask,dst). Native emit still succeeds and generates S.vmul(r2,r2,r3), changing x*z into z*z. Existing grouped memory operand checks do not cover Vmul. Therefore strict arithmetic correspondence is **NOT COMPLETE**, despite the valid compiled micro-case source and ABI protections.

Next repair, deferred while user diagnoses environment: extend existing MaterializationMap with one full actual Evaluate record per source_stmt and logical instance k, covering args/attributes/bindings for every non-sync operation (including Pset and Vmul). Strict native verification must independently walk final actual TIR and compare multiplicity-preserving full fingerprints against complete unique source-instance coverage, not a set or operation-kind list. Authored arithmetic changes followed by finalize must generate a changed valid map/source; post-finalize argument/attribute rewiring and extra/duplicate/missing effects must reject. Do not fix via comments or a separate sidecar authority. No production code changed for this follow-up.
