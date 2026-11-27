# Residual Convergence, Mechanism Sink-Down, and Gate Pyramid

Companion docs: [Architecture Overview](./00_architecture_overview.md) | [MECE Contracts and Routing](./01_mece_contracts_and_routing.md) | [Ascend Plugin README](../../targets/ascend/README.md)

## [SEC-G1] 3-Tier Cost Model Sink-Down & Sample-Gated Basis

Optimization insights follow a strict downward graduation path so structural slowdowns are caught statically without burning NPU leases:

1. **Stage-0 Discovery**: Real NPU profiling and BiSheng reports isolate a bottleneck in `ObservationVector5D`.
2. **Causal KB Hypothesis**: The fix starts as a falsifiable cross-operator rule in `seed_rules.yaml` gated by preconditions and `anti_precondition`.
3. **Mechanism Sink-Down**: Once structurally identifiable in lowered IR or compiler summaries, it sinks permanently into `CompilerBehaviorRegistry` and Tier-0/Tier-1 cost penalties.

`HybridAscendCostModel` (`ascend/cost_model/`) stacks three orthogonal tiers:

| Tier | Component | Deterministic Scope & Detectors |
| :--- | :--- | :--- |
| Tier-0 | Hardware Physics & 6-Pipe/AVE | Segmented HBM burst-bandwidth curves, vector/cube/scalar/MTE/FixPipe overlap, UB modes (`216 / 248 / 256 KiB`), wave occupancy, narrow unaligned & scalar-loop analytical penalties |
| Tier-1 | `CompilerBehaviorRegistry` (`_build_default_registry`) | 9 detectors: 7 baseline (`register_stack_spill`, `loop_vectorization_vloopv2`, `vf_local_membar`, `vf_fusion_breaks`, `alias_serialized_buffers`, `dual_issue_ipc_collapse`, `dcache_dcci_flushes`) + 2 Stage-6 sink-down (`scalar_compute_loop_fallback` covering TIR + reachable Track-B `__ubuf__`/`__gm__` scalar loops, `dcache_bypass_narrow_store`) |
| Tier-2 | `ResidualBasisRegistry` & `ExperienceLake` | `TopologyAxes`-bucketed Ridge log-residual calibration (`12` global `head0_specs`, `8` within-case `varies_within_case=True` `head2_specs` for zero-intercept pairwise delta ranking) |

`ResidualBasisRegistry` gates each basis term by minimum effective bucket samples (`2/4/6/8`) and quarantines extreme outliers:

| Sample Gate (`min_bucket_samples`) | `head0_specs` Features Unlocked (`12` total; `8` marked `*` in `head2_specs`) |
| :--- | :--- |
| `>= 2` | `intercept` |
| `>= 4` | `wave_tail_waste*`, `short_dma_burst_ind*`, `tier1_penalty_ratio*` |
| `>= 6` | `log_tier0_scale*`, `cube_bound_share*`, `vec_bound_share*`, `l2_miss_ratio*`, `launch_overhead_ratio*` |
| `>= 8` | `is_coupled_1c2v`, `is_aic_only`, `is_irregular_simt` |

- **Anomaly Guard (`|ln(m/a)| > 2.5`)**: Samples where `|ln(measured / analytical)| > 2.5` are logged to `unmodeled_anomalies.jsonl` and excluded from Ridge residual fitting; pairwise delta/log-residual targets are clamped to `[-2.5, 2.5]`.
- **Zero-Magic CostModel & Additive Latency Algebra**: `cost_model_mode` allows `off` (default), `shadow`, or `advisory` (with paired t-stat gating and zero heuristic floors). `latency_algebra.py` models additive microsecond conservation (`AnchorConservationPoint`, calibrated `dispatch_floor_us`, paired noise `sigma_paired_us`), while `FlywheelCoordinator` triggers distillation strictly when gain exceeds $3\sigma_{\text{paired}}$ and `tilesize_prescreen.py` prunes only physically infeasible configurations.

## [SEC-G2] Unified Compile Session & Anchor-Specialized Two-Snapshot Ledger

- **Unified `ascend_compile_session`**: Host lowering and driver build entrypoints (`build_stage_b_driver_source` and companions) must hold one contiguous `_ascend_ctx()` scope across `PyTorchSemanticAnalyzer`, anchor specialization, and `tilelang.lower` + `ReadOnlyAscendPassTap`. Exiting the target context prior to lowering strips hardware UB/L1 ceilings, vector lane widths, and target flags.
- **Pre-Lowering Anchor Specialization & `ArtifactStore`**: `EnvelopeInspector` (`ascend/schedule/envelope_inspector.py` in `L2`), `gate_funnel.py` + `stage_b_gate.py` (`agent/gates/` in `L2`), and `verifier.py` (`ascend/sil/` in `L0`) specialize dynamic-shape `PrimFunc` schedules via `PrimFunc.specialize` + `arith.Analyzer` before snapshot capture, while `ArtifactStore` (`ascend/compile_cache/artifact_store.py` in `L0`) indexes compiled `.so`/`.o` artifacts by canonical closure hash (`key = (source_sha, flags_sha, target, regime)`).
- **Preserved Two-Snapshot Diff**: Pre-lowering intended buffers and post-lowering physical buffers are compared side-by-side (`degraded_single_buffers`, lowered stage count, serialized barriers, `ObjectivePhysicalLedger`) across all canonical canary operators with zero operator-name branching.
- **Formal Legality & Direct `TypedFault` Emission**: `SILVerifier` enforces 14 hardware redlines plus C++ CFG cross-core flag-balance checks and Stage-A TileLang AST rules (`T.Stage`, `T.Parallel`); stages emit structured `TypedFault` records directly (`fault_types.py`).
- **CANN 9.3 Sole Active Mainline & No `syntax_repair` Heuristic KB**: Target `cann_9_3` (`dav-3510`) exclusively (`cann_9_3.toml`). `syntax_repair` runs only on compiler-error candidates (never `NOOP_IDENTICAL`), feeding objective compiler diagnostics plus the `G5` lock.

## [SEC-G3] Compilation Idempotence & Orthogonal Tail Guard

Every candidate promotion passes the 5-tier verification pyramid (`L0`–`L4`):

| Tier | Scope | Enforced Guarantees |
| :--- | :--- | :--- |
| `L0` | Static repo & AST audit | 6 repo hygiene lints (`scripts/*_lint.py`), `tests/ascend/l0_foundation/`, AST inequality whitelist, static rejection of float-tolerance or host-pointer keys in `_KERNEL_CACHE` |
| `L1` | Schedule & CFG legality | Parallel `run_l0_l1_round_filter` (`SILVerifier`, Schedule Redlines `G1..G5`, `ReadOnlyAscendPassTap`), `ArtifactStore` pre-compilation & `NOOP_IDENTICAL` short-circuit |
| `L2` | Lowered host envelope & Stage-B | Contiguous `ascend_compile_session`, Two-Snapshot diff, `StageBCompileGate` (`agent/gates/stage_b_gate.py`) quotient pre-compile, adaptive `TF_POISON_PILL_READY_FD` timeout, `explore_creative` pass-through |
| `L3` | Full contract, CEGAR & idempotence | Full shape suite (`l3_contract.py`), `PromotionCoordinator` ($3\sigma_{\text{paired}}$ non-regression + $2\sigma_{\text{paired}}$ gain + unbounded CEGAR guardrails), `L3Resolution` (`PROMOTED`/`CANDIDATE_REJECTED`/`INFRA_DEFERRED`), Two-Pass `Delta\|Cache\| == 0` |
| `L4` | Official real-NPU gate | Paired eval & CANNBench protocol (`agent/npu/l2_paired_eval.py`, `agent/npu/l3_l4_gates.py`, `tests/ascend/benchmarks/test_npu_acceptance.py`) via `AgentGates` leased NPU & signed v2 receipt with `device_fault` (`rc=75`) isolation |

- **Compilation Cache Idempotence Law**: `_KERNEL_CACHE` never uses heuristic capacity caps (e.g., `len(cases) // 2`) or float-tolerance/host-pointer keys. `L3` compiles twice under shape replay and dynamic outer-axis perturbation and requires `Delta|Cache| == 0`.
- **CEGAR Guardrails & Orthogonal Tail Guard**: `evaluate_promotion_gate` requires active worst anchor `>= 0.97x` within $3\sigma_{\text{paired}}$, all untouched siblings non-regressing within $3\sigma_{\text{paired}}$, net regime gain $\ge 2\sigma_{\text{paired}}$, and zero regression across all accumulated CEGAR `GuardrailCaseSpec` anchors.
- **`AgentGates` Execution Boundary & Fault Isolation**: `TileFoundry` owns contracts, passes, cost models, and search; `AgentGates` (`tilefoundry-npu-cpu` host / `tilefoundry-npu` NPU) owns Conda selection, NPU leasing, `device_fault` (`rc=75`) detection, and signed v2 receipts. Parent shells never export `TILEFOUNDRY_NPU_*`, and `INFRA_OCCUPANCY`/`INFRA_DEVICE_FAULT` never penalize the Bandit or stagnation counters.
