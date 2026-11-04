# P4 shared Z3 INPUT SPEC — PROPOSED, WBS 04-01 (I2)

Status: **SPECIFICATION ONLY. NOT AN IMPLEMENTATION, NOT A RECEIPT, NOT A DEVICE RESULT.**
Baseline: TF e73f0f2 / branch feat/tf-ascend950pr; AgentGates 4655144. CPU-only, no device.
This document defines the NORMALIZED INPUT to the shared Z3 core for the chain
"real emission → normalized Problem → shared solver". Every field below states (a) its
exact source: pass / file / line in this baseline, (b) its consumer, (c) its unit, and
(d) its refusal / error state. A diagram, three task names, a case-template tuple, or a
"certificate" is NOT an acceptable substitute for these fields.

The baseline implementation does NOT yet satisfy this spec: E09/E15 show
build_program() is a CASE-CONFIGURATION template with five fixed tasks and hard-coded
accesses/deps (program.py:128-147, problem.py:97-159), and solver.py:62-74 is a finite
Z3 certificate over three fixed loop tasks. Those are the current gaps, not the spec.
WBS 04-04/04-05 must replace them with a real emission ledger; 04-06/04-07 bind both
adapters to one physical pure-Z3 core. "PROPOSED" never means "already accepted".

## 0. Evidence vocabulary (HANDOFF §6, non-negotiable)

* An assertion counts only if the object under test produced it IN THE SAME RUN.
* No stale file results, no all(empty)==True, no "printed it" as "checked it".
* Every "cannot fail" test must be killed by a mutation (drop a constraint, add a
  fourth task, change a distance, drop the alias entry call, force z3 missing).
* Negatives must first prove they can fail; random search is not a non-existence proof.
* No ritual artifacts: a field that no consumer reads, a digest nobody verifies, or a
  status that never changes is a defect, not deliverable.

## 1. Dataflow and field ownership

| stage | pass / file (baseline lines) | emits | consumes |
|---|---|---|---|
| public finalize | src/tilefoundry/schedule/finalize.py:12-19 | BEFORE_LOWER → ConvertHIRToTIR → AFTER_LOWER Module | all later fields |
| typed HIR / emission ledger (PROPOSED 04-04) | src/tilefoundry/passes/transforms/convert_hir_to_tir.py:492-506 (StoragePlan), :1095-1140 (IssuePlan), :1230-1270 (GMEM alias view construction) | source/action stable IDs, actual accesses incl. implicit copies/carry | 04-05 Problem extractor |
| normalized Problem | targets/ascend/tilefoundry_ascend/schedule/problem.py:18-159 (current template; MUST be replaced) | ScheduleProblem fields of §2 | solver, sync, completion |
| shared Z3 core (PROPOSED 04-06/07) | targets/ascend/tilefoundry_ascend/schedule/solver.py:21-118 (current finite certificate) | SolveResult | plan |
| resolved plan | targets/ascend/tilefoundry_ascend/schedule/sync.py:42-80, passes/completion.py:29-39, pipeline.py:28-56 | obligations/stages/versions/plan_digest | TIR materialization, 05-xx |
| verification | targets/ascend/tilefoundry_ascend/lowering/verify.py, layout_check.py, placement.py:284-291 | typed reload coverage / capacity | architect review |

## 2. 输入来源与作废边界 (INPUT SOURCE AND VOID BOUNDARY) — NORMATIVE

This section turns the 03-21(b) VOID LIST (I2 VOID IDs V01–V10; these are NOT the
masterplan V01–V10 acceptance command cards) into hard constraints on EVERY Problem
field. Architect ruling: artifacts/run02/review/architect_ruling_03_21b_actual_inputs_run2.md.

| ID | constraint on Problem / solver input | named refusal |
|---|---|---|
| S-1 | No field may be derived from the pre-03-21 self-generated oracle inputs (numpy default_rng, ROW-WISE W of shape (rows,D), repeated 256-row blocks, ones/alternating w=1, dyn/recompute as two labels on one output). That "same-input / device-comparison / protocol-coverage" relation NEVER held; it is not a withdrawn approval. | SELF_GENERATED_INPUT_SOURCE |
| S-2 | Any input identity used by a Problem field must be runner ACTUAL X[M,D] / W[D] persisted by runtime_model.check_case (examples/rmsnorm_950pr/runtime_model.py:150-174, tensors.json convention at :225-231). W MUST be 1-D W[D]; a row-wise W is refused, never reshaped/broadcast. | ROW_WISE_W |
| S-3 | Each consumed input carries file sha256 (x/w/golden/device/tensors.json) plus a combined digest over exact BF16 codewords of X and W, with explicit shape header. A Problem without those digests is INCOMPLETE, not "unknown but fine". | INPUT_SHA_MISSING / INPUT_SHA_MISMATCH |
| S-4 | The frozen protocol is identified by protocol_id + protocol_sha256 + status FROZEN (artifacts/run01/p0/precision_protocol.json; capture_p0.py:55-61). A status-only check is insufficient: 04-10 must compare the pinned SHA/values externally. | PROTOCOL_NOT_FROZEN / PROTOCOL_SHA_MISMATCH |
| S-5 | "Same input" is an assertion, not a story: the runner FP64 golden recomputed from the persisted X/W must match the persisted golden.npy bit-exactly (rmsnorm_oracle.py identity_recompute / judge_pattern). Mismatch refuses the Problem's input binding. | INPUT_IDENTITY_FAILED |
| S-6 | Old VOID V01/V05/V09 (per-case stats, dyn==recompute, seed/pattern/shape fields as protocol coverage) CANNOT be used as loop-domain, guard or coverage evidence. 24 unique (case,key) pairs (4 exact cases x zero/ones/alternating/random_s0/s1/s2) must be enumerated; duplicates, missing keys and empty sets are refusals. | STALE_COVERAGE_CLAIM / EMPTY_CASE_SET |
| S-7 | V03/V06/V08/V10 narrowings are binding: BF16/single-RNE codec properties are NOT void (they are input-source-independent); the old 8192x4096 array WAS full-size (the defect was 32 repeated random blocks and runner identity, not shape); the frozen det-bits-exact obligation is NOT void; and there is NO blanket void of earlier comparisons — the run2 ACTUAL-INPUT independent audit (164 files / payload c8e21a39…, limited receipt) has no dependency on the old oracle and remains valid (not withdrawn, downgraded or upgraded). | VOID_OVERREACH |
| S-8 | Reference identities must stay separate: run2 audit FP64→FP32→BF16 vs new tool DIRECT FP64→BF16 RNE. Each keeps its own label; new max_abs 0.0078125 must NOT overwrite old 0.015625 or vice versa. | REFERENCE_IDENTITY_MIXED |
| S-9 | A tool PASS on an UNCOMMITTED source does not transfer credit to a new commit SHA: Problem.source_digest and manifest.tool.sha256 must bind the FINAL commit, with git status/dirty attribution (HANDOFF §6, architect 03-10 G01). | UNBOUND_SOURCE_SHA |
| S-10 | Model-only diagnostics (v33 chain model, vcadd PLACEHOLDER, UNSOURCED/UNVALIDATED native behaviour) are NOT pass criteria and MUST NOT be promoted into Problem correctness fields. | MODEL_AS_EVIDENCE |

## 3. Field catalogue (field → source → consumer → unit → error)

### 3.1 Identity and digests

| field | type / unit | source (pass / file / LINE) | consumer | error state |
|---|---|---|---|---|
| problem_id | string "case_id:digest16" | problem.py:157-159 (sha256 of program.as_dict(sort_keys=True)[:16]) | solver.request_digest, sync.build_plan, 05-01 plan | DUPLICATE_ID / STALE_PLAN (HIR change must change it) |
| source_digest | hex16, ONE payload domain | problem.py:157; capture_p0.py:138-142 (manifest OUTSIDE payload; whole-dir hash is a DIFFERENT domain) | SolveResult.request_digest (solver.py:38-39, :82-118), plan metadata | SOURCE_DIGEST_MISMATCH |
| target_profile | string | program.py:32 ("ascend950pr_9579"), problem.py:159 | facts resolution, both adapters | TARGET_PROFILE_UNKNOWN |
| policy_digest / facts_digest | hex sha256 over canonical policy + target facts | PROPOSED: policy from problem.py:144-150; facts from targets/ascend/tilefoundry_ascend/tf_target.py:91-98, src/tilefoundry/target/hardware/spec.py:31-55, tools/p0/capture_p0.py:96-106 (:101 topology, :106 execution_topology.json) | solver reproducibility, 05-01 plan metadata, 04-09 B↔C point 1 | FACTS_DIGEST_UNUSED (a digest nobody verifies is a ritual) |
| protocol binding | protocol_id + sha256 + FROZEN | artifacts/run01/p0/precision_protocol.json; capture_p0.py:55-61; solver currently checks NOTHING (load_protocol in the oracle checks status only) | 04-10 external check | PROTOCOL_SHA_MISMATCH |

### 3.2 Stable op / region / action IDs

| field | type / unit | source | consumer | error state |
|---|---|---|---|---|
| op_id | stable string, per authored HIR op | PROPOSED 04-04: convert_hir_to_tir.py:1095-1129 (IssuePlan.op selects the real instruction), :1131-1140 (plan/repeat/order); current problem.py:101-114 uses FIVE hard-coded task ids from case config | 04-05 extractor, 05-06 verify_tir coverage | ID_FROM_CASE_TEMPLATE / ID_FROM_id(obj) (cross-process instability) |
| region_id | stable string (e.g. weight_convert / main_compute), one per SimdVF region instance | program.py:77-86 VFRegion, execution.py:43-58 (nestable=False; same region INSTANCE required for mask dominance), convert_hir_to_tir.py:499-506 (ValuationVisitor contexts) | dominance checks, 04-05 grouping | DIFFERENT_REGION_INSTANCE / LEXICALLY_LATE_PRODUCER |
| action_id | stable string per emitted access / implicit copy / carry | PROPOSED 04-04: StoragePlan.answers (convert_hir_to_tir.py:508-520) and IssuePlan.plan (:1131-1140); ALL actual lowering actions, including insert_slice/carry | bijection with Problem.accesses/tasks | UNKNOWN_ACTION / MISSING_IMPLICIT_COPY |
| task.stable_id | string | problem.py:37-48 Task, :101-114 (currently fixed: w_load/w_convert/x_load/main_vf/y_store) | solver.py:21-22 LOOP_TASKS/EDGES_RAW, sync.py:42-57, materialize | DUPLICATE_TASK / EXTRA_TASK_IGNORED (a fourth task MUST change solving) |

ID rules: derive from source/action emission order + a canonical string, NEVER from
Python id(obj)/hash(), NEVER from the case name alone, NEVER from a three-task list;
stable across processes and across flag renames in the B↔C correspondence (04-09).

### 3.3 Loop domain, guards and logical addresses

| field | type / unit | source | consumer | error state |
|---|---|---|---|---|
| loop_domain.start | expression string | program.py:35-49 (tiles/nbase/rem/iters/loop_stop), problem.py:152-155 | solver stage count, 03-24 core start | CORE_START_OFF_BY_ONE |
| loop_domain.count_expr / count_dyn / count_recompute | expression / integer loop iterations | program.py:47-53, problem.py:153, run_all_cases.sh:100-103 (four exact case params) | solver.py:77-103, sync.build_plan, materialize TIR guards | RAGGED_TAIL_MISSING / LOOP_MODE_COLLAPSED |
| loop_domain.nbase / rem / tiles | integers (tiles) | program.py:35-45, problem.py:154 | 56-AIV ownership (32x37+24x36 / 8x2+48x1) | OWNER_COLLISION |
| loop_domain.induction_id | string "step" | problem.py:155; generation is the LOGICAL iteration ordinal t, NOT the recompute tile index (sync.py:3-6) | 05-02 generation/slot | TILE_INDEX_AS_GENERATION |
| loop_domain.guards | list of guard expressions; CURRENTLY [] (problem.py:155) — an EMPTY list must not be reported as "all guards checked" | PROPOSED: dynamic boundaries, ragged tail, prologue/steady/drain (03-24, 05-05) | materialize_tir guard placement | EMPTY_GUARD_SET_CLAIMED / SHORT_ITERATION_OOB |
| logical address | (root, byte_offset, extent, dtype, access, logical_iteration) | dtype_view.py:44-78 (DtypeViewSpec), convert_hir_to_tir.py:1234-1258 (GMEM alias view: byte_offset=0, access="read" TODAY — MUST NOT be generalised), problem.py:18-26 Access.logical_iteration ("t" / "outside") | alias_conflicts (view_spec.py:104-123), 05-03 physical address | UNKNOWN_ADDRESS / OFFSET_ASSUMED_ZERO |
| byte_offset / count | BYTES / target-dtype ELEMENTS | view_spec.py:32 (DTYPE_WIDTH), :69-82 _tags; dtype_view.py:59-65 nbytes/end_byte | alias + placement | VIEW_MISALIGNED_OFFSET / VIEW_RANGE_OVERFLOWS_PARENT |

### 3.4 Tasks, resources, read/write footprints, alias class

| field | type / unit | source | consumer | error state |
|---|---|---|---|---|
| task.pipe / task.resource | enum "mte2"/"vector"/"mte3" | problem.py:37-47, :101-114; ISA resource isa.py:74-95 | solver._resource_lb (solver.py:42-47), sync pipes | UNKNOWN_PIPE |
| task.latency / issue_interval | MODEL cycles (NOT device time, NOT measured) | problem.py:89-94 (_latency_vector/_ii_resource from ISA.latency_model/ii_model, isa.py:80-95), :109-114 | solver._ii_for (solver.py:50-59) | MODEL_II_AS_DEVICE_LATENCY |
| access.task/root/space/mode/byte_footprint/logical_iteration | mode read\|write, footprint BYTES | problem.py:18-26, :116-129 (xb = rows*d*2 bytes; c.d*2 / c.d*4) | placement.classify (placement.py:131-160), capacity_check (pipeline.py:16-21), alias_conflicts | ACCESS_MODE_MISSING (a missing access MUST NOT default to read) / NEGATIVE_FOOTPRINT |
| capacity (nominal/usable/known) | BYTES, per AIV | memory.py SPACE_FACTS (:66), placement.py:48-62 SpaceSemantics, :224-281 place(), :284-291 require_known_capacity, pipeline.py:16-21 | solver feasibility, 05-03 placement | CAPACITY_UNSOURCED / UB_OVERFLOW / REGISTER_WIDTH_AS_TOTAL_BUDGET |
| alias class | (root, BYTE range); dtype EXCLUDED from identity | view_spec.py:85-110 alias_conflicts (UNIQUE entry), :112-130 check delegates only to it; dtype_view.py:102-115 access-based overlap | 04-02 plan-level generation/cap, 05-07 physical hazards | VIEW_OVERLAP_INCOMPATIBLE (any write on intersecting same-root ranges), ALIAS_ENTRY_BYPASSED (deleting the call must kill the negative), DTYPE_AS_ALIAS_IDENTITY |

Rules: read/read overlap is ALLOWED, any write on an intersecting same-root range is
FORBIDDEN regardless of dtypes; staleness (generation) and element-cap are spec/plan
layer, not target-layer (dtype_view.py:21-24 scope limits). The unique alias entry has
a pinned mutation test: tools/p3/test_view_spec_single_entry.py.

### 3.5 Dependencies, versions, capacity / owner, policy

| field | type / unit | source | consumer | error state |
|---|---|---|---|---|
| dependency.producer/consumer/kind/distance/evidence | distance in LOGICAL ITERATIONS; kind RAW\|WAR\|WAW\|reuse | problem.py:28-34, :131-139 (X free +Bx, Y free +By, weight live-in, recompute WAW distance 1) | solver._z3_certificate (solver.py:62-74), sync.build_obligations (sync.py:42-57) | DEPENDENCY_DISTANCE_CHANGED_IGNORED / RECOMPUTE_WAW_DROPPED |
| buffer.root/space/bytes_/alignment/owner_loop/versions/fixed | bytes, slots, fixed bool | program.py:62-69 BufferDecl, :134-141 (x_ub/y_ub ring versions, w_ub/wf fixed, GM roots), problem.py:51-59, :141-142 | solver.py:77-103 (S, Vx, Vy), sync.build_plan (:71-80), 05-02 owner/generation | VERSION_BELOW_ONE / STAGE_BUDGET_AS_BUFFER_VERSION / NESTED_CHUNK_STOLE_OWNER |
| policy.requested_num_stages / offset_policy / effective_offset / x_versions / y_versions / loop_mode | stages (dimensionless), bool, slots, enum | program.py:88-92 PipelinePolicy, :146; problem.py:144-150 (source_default → False, backend default); run_all_cases.sh:100-103 | solver.py:77-118, 04-03 parse/transform/print/reload retention | POLICY_LOST_IN_PASS / OFFSET_DEFAULT_TRUE / STAGE_COUNT_AS_NONEMPTY_STAGES |
| policy + facts digest | canonical JSON sha256 | PROPOSED from problem.py:144-150 + tf_target.py:91-98 + hardware/spec.py:31-55 + capture_p0.py:138-142 | 04-09 B↔C, 05-01 plan metadata | DIGEST_UNUSED / PAYLOAD_VS_WHOLEDIR_MIXED |

### 3.6 Error states (sat / unsat / unknown / fallback) — MUST remain four distinct values

| state | meaning | source / current gap | consumer behaviour |
|---|---|---|---|
| sat | Z3 proved feasibility of the returned stage/II assignment | solver.py:62-74 _z3_certificate returns s.check()==z3.sat (certificate only); full core PROPOSED | return SolveResult("solved", ...) ONLY with a real sat on the FULL constraint set; a "certificate" over three hard-coded tasks is NOT sat for the real Problem |
| unsat | constraints infeasible (S<1, Vx/Vy<1, no stage assignment) | solver.py:80-86, :101-103 | NO plan, NO TIR, NO fallback hidden as solved; record diagnostics |
| unknown | solver could not decide, OR z3 rejected the chosen assignment | solver.py:64-65 (z3 None), :107-110 (cert False) | NO solved claim. Current defect: z3 missing still allows "solved" (E10); 04-08 must kill that mutation |
| fallback | explicit serial/single-stage plan chosen by policy, with its own status | NOT IMPLEMENTED today (solver.py has no fallback status); pipeline.py:39-42 returns a NOT-SOLVED structure | MUST be reported as fallback with reason; never presented as a multi-stage schedule or as sat |

Additional refusals (not solver states): SELF_GENERATED_INPUT_SOURCE, ROW_WISE_W,
INPUT_SHA_MISMATCH, PROTOCOL_SHA_MISMATCH, INPUT_IDENTITY_FAILED, EMPTY_CASE_SET,
EMPTY_GUARD_SET_CLAIMED, MODEL_AS_EVIDENCE, VOID_OVERREACH, UNKNOWN_ACTION,
CAPACITY_UNSOURCED, ALIAS_ENTRY_BYPASSED. Each needs a negative that first proves it
can fail and a mutation that kills the corresponding check.

### 3.7 Solver output fields that the resolved plan consumes (04-05/05-01 boundary)

| field | type / unit | source | consumer | error state |
|---|---|---|---|---|
| SolveResult.status / ii / stages / order | enum, model cycles, dimensionless stage ordinals | solver.py:26-39 SolveResult, :77-118 solve() | sync.build_plan (sync.py:71-80), pipeline.py:36-56 | STATUS_COLLAPSED / N_STAGES_AS_NONEMPTY_STAGES |
| SolveResult.request_digest / model_version / solver_version / seed | hex16 / string / string / integer | solver.py:23 MODEL_VERSION, :32-35, :82-118; pipeline.py:56 plan_digest | 04-08 reproducibility, 05-01 plan metadata | STALE_PLAN / SEED_UNPINNED |
| SolveResult.diagnostics | list of strings | solver.py:80-118 (resource_lb, effective_offset, z3 certificate) | 04-08 mutation feedback, architect 04-10 | Z3_MISSING_REPORTED_SOLVED |

## 4. Units (single table, no mixing)

| quantity | unit | forbidden interpretation |
|---|---|---|
| byte_offset / byte_footprint / bytes_ / capacity | BYTES | element count; 256 B register WIDTH as aggregate register budget (tf_target.py:72-78) |
| count / extent | ELEMENTS of the viewed dtype | bytes |
| distance / stages / slots / versions | LOGICAL ITERATIONS / dimensionless ordinals | wall-clock time or device latency |
| latency / issue_interval | MODEL CYCLES | measured NPU time; a Z3 sat is NOT hardware completion |
| digest / ID | canonical string (hex or stable ID) | a hash of the whole directory when the payload domain was meant (capture_p0.py:138-142; ARCHIVE-DOMAIN.md) |
| atol / rtol / eps | frozen protocol scalars | adjustable thresholds; max_abs<=atol is NEVER the pass rule (HANDOFF §5.9) |

## 5. The three easiest places to turn this spec into RITUAL

1. **Stable IDs from the case template.** Copying the five task names from
   program.py:101-114 (or three names from solver.py:21) looks like an "ID column" but
   has no bijection with real emission actions; deleting a store/adding a copy would not
   change it. Kill: 04-04 mutation must change the ledger AND the Problem.
2. **A facts/policy digest nobody consumes.** Writing sha256 into the manifest while
   solver uses only policy.requested_num_stages: a digest that cannot invalidate a stale
   plan is a ceremony. Kill: source/facts/protocol digest change must refuse the old plan.
3. **Collapsing unknown/fallback into solved.** solver.py:64-65 / :107-118 currently lets
   a missing z3 or a finite certificate still report "solved"; a status field that never
   changes is decoration. Kill: force z3 absent and assert status != solved; add a fourth
   task / change a distance and assert the result changes or refuses.

## 6. What 04-01 does NOT claim

No shared core exists yet, no TF/TL adapter pair, no real emission ledger, no B↔C
correspondence, no device measurement, no P4 credit. This is the reviewed INPUT CONTRACT
that 04-02..04-10 must implement and that architect 04-10 will re-extract.
