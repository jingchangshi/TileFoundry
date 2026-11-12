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
