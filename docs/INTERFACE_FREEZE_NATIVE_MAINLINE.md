# Interface freeze — TileFoundry Ascend 950PR native mainline

本记录冻结当前单一主链接口；只允许一个 typed schema 与一条生产路径，不提供版本分派、adapter、wrapper、fallback 或历史测试对齐分支。
> 合同签名：`sha256:a6764633207d175148bcfdbdb6ff2e24eab006d31ac5b8d135f3f90d5752f790`；本文件是唯一 API 规范。

## 数据结构
- PipelineSpec：单一 typed schema。字段为 max_stage, fixed_ii | None, max_buffer_versions, enable_offset, task_intents: tuple[TaskIntent], buffer_schedules: tuple[BufferSchedule]。max_stage 是 stage 上界；max_buffer_versions 是未单独声明 stream buffer 的 auto 版本上界；二者独立。Fixed(V) 只能来自 BufferSchedule。
- TaskIntent：task_id, anchor_ref（真实 direct-child 对象）, stage, issue_after, cost_override, source_ref。
- BufferSchedule：root_ref（真实 allocation/buffer 对象）, versions=Fixed(n) | Auto(min,max), owner, space, layout。
- RegionBindingSummary：source LoopRegion、captures/carried params/initial values/yields、lowered For/Var/alloc refs，以对象 identity 绑定。
- UnitTree：whole-task Unit，含 node_ref、region_ref、bounds(start/stop/step Expr)、effects、owner、resources、predicate、captures/carries/yields；字符串 path 仅打印。
- NormalizedProblem：全部 source task/effect/access/buffer/resource/owner/pipe/stage-order，没有 source-only sidecar；无法表达的域在 problem 构造阶段具名拒绝。
- ResolvedPlan：唯一 authority；RegionPlan 有 native region ref、lower/stop/step Expr、II、origin、max_shift、tasks(stage/phase/start/issue_rank)、buffers(fixed/auto domain/resolved_versions)；completion/storage/entry 全部消费同一版本字段。ResolvedPlan schema 唯一写者为 MAIN-A/implementer-g2；Lead 只裁定与集成，不并行修改 schema。
- Deadline：`solve_region(problem, deadline)` 的 deadline 是共享进程 monotonic 绝对截止时间（float 秒）；`None` 仅表示显式无截止。任何阶段传递不得重置、延长或重新开始预算。
- SymbolicCompletionPlan：有限 CompletionSite/Channel，dense FREE，每代 Set/Wait，尾 min(V,N)，真实 PhysicalEventNamespace；不按 N 展开。构建入口必须消费 effects。
- StoragePlan：PhysicalLayout/Requirement/StorageKey/LiveWindow，versions 后分配，scratch 单次计费，capacity 来自当前 Target facts；UNKNOWN_CAPACITY 拒绝而不是回填。
- NativeArtifact：TL source、parsed PrimFunc expected facts、ABI、source/plan/TIR/TL identity、pass_configs（tl.enable_auto_schedule=false, tl.disable_shared_memory_reuse=true）、target/toolchain identity。

## 职责 API（沿用现有模块，不新建框架）
- normalize_intent(native HIR) -> typed intent + RegionBindingSummary
- build_native_unit_tree(checked TIR, region_bindings) -> UnitTree
- analyze_native_problem(units, effects, alias, target) -> NormalizedProblem + source coverage verdict
- solve_region(problem, deadline: float | None) -> Assignment(status SAT/UNSAT/UNKNOWN, starts, stages, phases, versions, issue_order, optimality); deadline is a shared process-monotonic absolute deadline in seconds; None means explicitly unbounded; no stage may reset it
- resolve_schedule(problem, assignment) -> ResolvedPlan
- build_symbolic_completion(plan, target, effects) -> SymbolicCompletionPlan
- place_resolved_versions(plan, completion, target, lowering) -> StoragePlan
- apply_symbolic_pipeline(entry, plan, completion, storage) -> private candidate module + source_map
- verify_actual_program(original, rewritten, plan, completion, storage) -> verdict
- install_entry(module, rewritten) -> module (actual Module.functions replacement; only after full VERIFIED)
- emit_supported_tl(module, plan, completion, storage, target) -> NativeArtifact
- read_actual_tl(artifact, parsed_prim_func, refinement_manifest) -> conformance verdict
- bind_invoke_once(artifact, abi, inputs, device) -> outputs (one invocation; no host math)

## 编译事务
1. freeze immutable source snapshot before materialization.
2. analyze/solve/resolve once per plan iteration using the shared process-monotonic absolute deadline; no stage may reset it.
3. completion -> storage -> `apply_symbolic_pipeline` -> private candidate module plus source_map. The private candidate is not installed, exported, or bound.
4. `emit_supported_tl` plus `read_actual_tl` (parsed PrimFunc, refinement manifest, dual PassConfig identity, source/plan/TIR/TL identity) must yield a complete VERIFIED verdict on the private candidate.
5. `read_actual_tl` conformance must pass before any `install_entry`, public export, or bind. Only the full VERIFIED candidate may replace actual `Module.functions`.
6. any failure keeps the original module/entry, refuses export, and publishes nothing; no metadata-only APPLIED.
7. explicit TL compile uses both PassConfig keys; TL AutoSchedule/refinement must not change the ResolvedPlan.
8. local commit/export only after actual entry, plan/completion/storage identity, private candidate conformance, and install preconditions are rechecked.

## 写者边界
- MAIN-A/implementer-g2：ResolvedPlan schema 唯一写者；Lead 只裁定与集成，不并行修改 schema。
- MAIN-B/flex-1-g2：completion/storage/TL/runtime/AR/MLA native builders 后半链。
- 其他角色只读复审/验证；不写生产文件。

## 验证边界
- N=0/1/V-1/V/V+1/17/1_000_003 actual IR structure constant except constants.
- Two distinct inputs produce non-trivial correct CPU evaluator result.
- Wrong operand, dropped task/dependency, wrong slot/generation/owner, missing writer wait/tail wait, stale layout offset, stale source/plan/ABI identity all refused.
- Original 20+20 contract, cold/hit, device timing, dispatch, independent reproduction remain final gates.
