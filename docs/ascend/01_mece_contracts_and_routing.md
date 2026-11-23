# MECE Contracts, Compilation Session, and Regime Routing

Companion docs: [Architecture Overview](./00_architecture_overview.md) | [Residual Convergence and Gates](./02_residual_convergence_and_gates.md) | [Ascend Plugin README](../../targets/ascend/README.md)

## [SEC-C1] Problem Space & Bounded 2-4 Regimes

Every candidate kernel maps to one cell in `TopologyAxes` (`3 x 6 x 4 = 72` orthogonal cells) used to isolate empirical calibration across `ExperienceLake` and `HybridAscendCostModel`:

| Axis | Constant | Values |
| :--- | :--- | :--- |
| `compute_engine` | `VALID_COMPUTE_ENGINES` (3) | `aiv_only`, `aic_only`, `coupled_1c2v` |
| `memory_ring` | `VALID_MEMORY_RINGS` (6) | `gm_ub_stream`, `ub_random_atomic`, `gm_l1_l0_fixpipe`, `l0c_ub_epilogue`, `bidirectional_cv_ring`, `persistent_onchip_state` |
| `iteration_regularity` | `VALID_ITERATION_REGULARITIES` (4) | `affine_simd`, `irregular_simt`, `ragged_group`, `dynamic_opcode_stream` |

Multi-shape dispatch (`ascend/sil/`, `pytorch_frontend.py`, `regime_solver.py`) classifies cases purely from physical ratios (zero operator-name branching):

- **`CasePhysicsRatios` & 5D `PhaseSignature`**: Computes `stride_ratio`, `contig_atom_ratio`, `is_atom_aligned`, `working_set_sram_ratio`, `working_set_vreg_ratio`, `wave_occupancy_ratio`, `oi_ridge_ratio`, `inner_contiguous_bytes`, and `has_empty_group`, projecting to `phase_signature = (engine, layout, residency, alignment, wave)`.
- **Sub-Atom Separation**: When `contig_atom_ratio < 1.0`, reduction cases receive the distinct `_subatom` tag (`sub_atom_simt` alignment band) so sub-atom reductions never merge silently with wide aligned vector reductions.
- **Weighted Lexicographic `2..4` Bounded Partition (`partition_case_regimes`)**: Enforces `1 <= min_partitions <= max_partitions` (default `min_partitions=2`, `max_partitions=4`) producing `CasePhysicsRegimeGroup` buckets. Excess clusters merge by minimum 5D `_phase_distance` (`engine` +100, `layout` +30, `sub_atom_simt` alignment +20 vs +3, `residency` +2, `wave` +1); single-cluster inputs split along `_PHASE_SPLIT_AXES = ("wave", "residency", "alignment", "layout", "engine")` or by dtype-aware `case_workload_scale`.
- **SIL Formal Legality**: `OpAdvisoryExtractor` emits `AdvisoryEnvelope`; `SILVerifier` enforces 14 hardware redlines plus C++ CFG cross-core flag-balance invariants.

## [SEC-C2] Action Lattice, 3-Layer Prompt & COW Isolation

Legal transformation freedom is partitioned into the 7-facet `ConstraintRelaxationMask` (`VALID_FREEDOM_FACETS`):

| Facet | Default | Physical Scope |
| :--- | :--- | :--- |
| `alg_compute_and_monoid` | Enabled | Associative reduction rewrites, online softmax/norm monoid fusion, numeric-safe algebraic equivalents |
| `tile_and_stage_knobs` | Enabled | Tile shapes, split factors, pipeline stage depth, multi-core block mapping |
| `exec_mode_and_spr` | Locked | Vector/SIMT execution mode switches, UB capacity modes (`216 / 248 / 256 KiB`), SPR flags |
| `sram_layout_and_aliasing` | Locked | UB/L1 swizzling, stride padding, non-aliasing scratchpad buffer overlay |
| `dma_and_cache_hierarchy` | Locked | Burst coalescing, GM-to-L1-to-UB staging, L2 locality hints, cache bypass avoidance |
| `pipeline_and_sync_graph` | Locked | Cross-pipe event allocation, software pipelining, ping-pong double buffering, flag balance |
| `grid_and_workspace_topology` | Locked | Asymmetric core mapping, dynamic tail balancing, GM workspace partitioning |

- **Dual-Track Routing (`CapabilityGate`)**: Combines 3-axis `(TopologyAxes, PipeGapReport, CasePhysicsRatios)` routing with `ISAGapScanner` to select `tilelang_dsl` vs `ascendc_template` (`Router=ON/OFF` ablatable) across `seed_synthesizer.py`, `portfolio_runner.py`, and `campaign_runner.py`.
- **Unified 3-Layer Prompt (`llm_constraints.py`, zero Layer 1.5)**:
  - **L1 (`format_hardware_and_sil_layer`)**: DaVinci950 memory caps, 14 `SILVerifier` redlines, Z3 modulo pipeline bounds, 7-facet lock state, and universal Schedule Integrity Redlines `G1..G4`—each paired with its canonical escape path (`G1` vector/matrix over scalar loops, `G2` `>=32B` burst packing over `<32B` DCache-bypass stores, `G3` `T.dynamic` + 2D `T.copy` over key pollution/unrolled copies, `G4` disjoint `coupled_1c2v` subcore GM slices).
  - **L2 (`format_target_dialect_layer`)**: Active `cann_9_3` `dav-3510` API surface (`cann_9_2.toml` is a frozen inert placeholder—target `cann_9_3` exclusively with zero `cann_9_2` conditional shims), two-stage on-demand pruning, 4 lowering guardrails, and `seed_rules.yaml` entries guarded by `anti_precondition` and `l2_pmu_available`.
  - **L3 (Causal Feedback)**: Top-ranked KB hypotheses matched to the active `SymptomFamily` and 3-axis Bayesian utility.
- **`syntax_repair` & `explore_creative` Discipline**: Syntax repair and Stage-0 retries apply `L1 + L2 + G5` Parent Monotonicity Lock (`format_g5_parent_monotonicity_lock`, forbidding dropping `T.Pipelined` stages or downgrading execution domains). `explore_creative` remains `Rule-Blind` to L3 heuristics, gated by L1 legality, compilation, and bounded analytical pass-through (`predicted_us <= 1.5 * parent_analytical_us`).
- **CoW Helper Isolation (`patch_regime_into_module`)**: Clones/prefixes regime-local helpers as `_{regime_tag}_*` instead of mutating shared helpers in place, prunes unreachable dead helpers, and enforces the dual-layer orthogonal tail guard (`>= 0.97x` worst case in active regime; `>= 0.98x` untouched sibling regimes with single-point anti-jitter recheck).

## [SEC-C3] 5D Observation, 13 Symptom Families & Frozen KB

`ObservationVector5D` and `AxisResidual5D` (`observation_space.py`, `msprof_report_parser.py`) bridge L0 physical ledger, L1 BiSheng compiler summary, and L2 MSProf telemetry across 5 orthogonal axes (`compute_issue`, `offchip_dma`, `onchip_sram_reg`, `pipeline_sync`, `grid_irregularity`/`grid_wave`):

- **`transfer_inflation_ratio`**: `measured_hbm_bytes / max(1.0, blueprint_min_gm_bytes)` when measured HBM bytes exist, else `(blueprint_min_gm_bytes + static_extra_roundtrip_bytes) / max(1.0, blueprint_min_gm_bytes)`.
- **`effective_compute_eta`**: `blueprint_compute_floor_us / max(1e-3, measured_latency_us)`.

Every diagnosed waste maps deterministically to the closed 13-member `SymptomFamily` taxonomy (`agent/kb/schema.py`):

| `SymptomFamily` (13 Closed Members) | `SYMPTOM_FAMILY_TO_AXIS` | `SYMPTOM_FAMILY_TO_FACETS` |
| :--- | :--- | :--- |
| `DMA_REDUNDANT_GM_ROUNDTRIP` | `offchip_dma` | `dma_and_cache_hierarchy`, `sram_layout_and_aliasing`, `pipeline_and_sync_graph` |
| `DMA_UNALIGNED_SHORT_BURST` | `offchip_dma` | `tile_and_stage_knobs`, `dma_and_cache_hierarchy` |
| `DMA_L2_THRASH_REUSE_MISS` | `offchip_dma` | `grid_and_workspace_topology`, `dma_and_cache_hierarchy` |
| `COMPUTE_VECTOR_LOOP_DEGRADED` | `compute_issue` | `exec_mode_and_spr`, `alg_compute_and_monoid` |
| `COMPUTE_INEFFECTIVE_THROUGHPUT` | `compute_issue` | `exec_mode_and_spr`, `tile_and_stage_knobs` |
| `COMPUTE_FIXPIPE_ACCUM_IMBALANCE` | `compute_issue` | `tile_and_stage_knobs`, `sram_layout_and_aliasing` |
| `COMPUTE_1C2V_SUBCORE_IMBALANCE` | `compute_issue` | `grid_and_workspace_topology`, `pipeline_and_sync_graph` |
| `SCALAR_LOOP_TRAP` | `compute_issue` | `exec_mode_and_spr`, `dma_and_cache_hierarchy` |
| `SRAM_REGISTER_SPILL_STACK` | `onchip_sram_reg` | `tile_and_stage_knobs`, `exec_mode_and_spr` |
| `SRAM_UB_L1_BANK_CONFLICT` | `onchip_sram_reg` | `sram_layout_and_aliasing` |
| `SYNC_PSEUDO_DOUBLE_BUFFER` | `pipeline_sync` | `pipeline_and_sync_graph`, `sram_layout_and_aliasing` |
| `SYNC_BARRIER_SERIALIZATION` | `pipeline_sync` | `pipeline_and_sync_graph`, `tile_and_stage_knobs` |
| `GRID_TAIL_WAVE_UNDERUTIL` | `grid_wave` | `grid_and_workspace_topology`, `tile_and_stage_knobs` |

- **Frozen Online KB (`agent/kb/`)**: `seed_rules.yaml` holds 41 cross-operator causal rules (`R01..R41`) evaluated read-only online under a strict AST inequality whitelist (`_ALLOWED_AST_NODES`), ranked by 3-axis factorized Bayesian tracking (`ce:*`, `mr:*`, `ir:*`), and pruned against repeated dead-ends via Scoped Tabu (`excluded_mechanisms`).
