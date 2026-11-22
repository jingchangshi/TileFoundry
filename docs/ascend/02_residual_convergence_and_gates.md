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
| Tier-1 | `CompilerBehaviorRegistry` (`_build_default_registry`) | 7 baseline detectors (`register_stack_spill`, `loop_vectorization_vloopv2`, `vf_local_membar`, `vf_fusion_breaks`, `alias_serialized_buffers`, `dual_issue_ipc_collapse`, `dcache_dcci_flushes`) + 2 Stage-6 sink-down detectors (`scalar_compute_loop_fallback`, `dcache_bypass_narrow_store`) |
| Tier-2 | `ResidualBasisRegistry` & `ExperienceLake` | `TopologyAxes`-bucketed Ridge log-residual calibration (`12` global `head0_specs`, `8` within-case `varies_within_case=True` `head2_specs` for zero-intercept pairwise delta ranking) |

`ResidualBasisRegistry` gates each basis term by minimum effective bucket samples (`2/4/6/8`) and quarantines extreme outliers:

| Sample Gate (`min_bucket_samples`) | `head0_specs` Features Unlocked (`12` total; `8` marked `*` in `head2_specs`) |
| :--- | :--- |
| `>= 2` | `intercept` |
| `>= 4` | `wave_tail_waste*`, `short_dma_burst_ind*`, `tier1_penalty_ratio*` |
| `>= 6` | `log_tier0_scale*`, `cube_bound_share*`, `vec_bound_share*`, `l2_miss_ratio*`, `launch_overhead_ratio*` |
| `>= 8` | `is_coupled_1c2v`, `is_aic_only`, `is_irregular_simt` |

- **Anomaly Guard (`|ln(m/a)| > 2.5`)**: Samples where `|ln(measured / analytical)| > 2.5` are logged to `unmodeled_anomalies.jsonl` and excluded from Ridge residual fitting; pairwise delta/log-residual targets are clamped to `[-2.5, 2.5]`.

## [SEC-G2] Unified Compile Session & Anchor-Specialized Two-Snapshot Ledger

- **Unified `ascend_compile_session`**: Host lowering and driver build entrypoints (`build_stage_b_driver_source` and companions) must hold one contiguous `_ascend_ctx()` scope across HIR-to-TIR binding, anchor specialization, and `tilelang.lower`. Exiting the target context prior to lowering strips hardware UB/L1 ceilings, vector lane widths, and target flags.
- **Pre-Lowering Anchor Specialization**: `EnvelopeInspector` (`envelope_inspector.py`, `gate_funnel.py`, `verifier.py`) specializes dynamic-shape `PrimFunc` schedules with real anchor dimensions via `PrimFunc.specialize` and simplifies bounds with `arith.Analyzer` *before* snapshot capture so peak UB/L1 bytes, double-buffering, and contiguous burst widths reflect physical reality.
- **Preserved Two-Snapshot Diff**: Pre-lowering intended buffers and post-lowering physical buffers are compared side-by-side (`degraded_single_buffers`, lowered stage count, serialized barriers, `ObjectivePhysicalLedger`) across all canonical canary operators with zero operator-name branching.
- **Formal Legality**: `SILVerifier` enforces 14 hardware redlines plus C++ CFG cross-core flag-balance checks; `ValidateAscendPipelines` / `ResolveAscendPipelines` solve modulo pipeline schedules via Z3.

## [SEC-G3] Compilation Idempotence & Orthogonal Tail Guard

Every candidate promotion passes the 5-tier verification pyramid (`L0`–`L4`):

| Tier | Scope | Enforced Guarantees |
| :--- | :--- | :--- |
| `L0` | Static repo & AST audit | 6 repo hygiene lints (`scripts/*_lint.py`), AST inequality whitelist, static rejection of float-tolerance or host-pointer keys in `_KERNEL_CACHE` |
| `L1` | Schedule & CFG legality | `SILVerifier` (14 redlines + C++ cross-core CFG flag balance), Schedule Redlines `G1..G5`, Z3 pipeline resolution |
| `L2` | Lowered host envelope | Contiguous `ascend_compile_session`, real-anchor Two-Snapshot diff, Tier-0/1/2 sanity & bounded `explore_creative` pass-through (`<= 1.5x`) |
| `L3` | Full contract & idempotence | Full operator shape suite (`l3_contract.py`), Dual-Layer Tail Guard, Two-Pass Compilation Idempotence (`Delta\|Cache\| == 0`) |
| `L4` | Official real-NPU gate | CANNBench protocol correctness & latency (`surrogate_cannbench.py`, `test_npu_acceptance.py`) via `AgentGates` leased NPU & signed v2 receipt |

- **Compilation Cache Idempotence Law**: `_KERNEL_CACHE` never uses heuristic capacity caps (e.g., `len(cases) // 2`) or float-tolerance/host-pointer keys. `L3` compiles twice under shape replay and dynamic outer-axis perturbation and requires `Delta|Cache| == 0`.
- **Dual-Layer Orthogonal Tail Guard**: Active regime worst benchmarked anchor must remain `>= 0.97x` baseline; all untouched sibling regimes must remain `>= 0.98x` with automatic single-point anti-jitter recheck.
- **`AgentGates` Execution Boundary**: `TileFoundry` owns contracts, passes, cost models, and search; `AgentGates` (`tilefoundry-npu-cpu` host / `tilefoundry-npu` NPU) owns Conda selection, NPU leasing, timeouts, and signed v2 receipts. Parent shells must never export `TILEFOUNDRY_NPU_*`, and no `.venv`/`venv` may exist inside the repo.
