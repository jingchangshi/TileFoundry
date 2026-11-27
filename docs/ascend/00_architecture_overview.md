> Audience: Human / First-Time Architect Overview. Coding subtasks DO NOT read this file; follow section-sliced routes in AGENTS.md.

# TileFoundry Ascend End-to-End Architecture Overview

TileFoundry Ascend couples deterministic symbolic compilation with dual-track neuro-symbolic schedule evolution for DaVinci950 (`dav-3510`, `cann_9_3` sole active line, `Ascend950PR` / `Ascend950DT`, backed by unified `davinci_950.toml` + `cann_9_3.toml`) across a strict lower-triangular 6-layer DSM (`L0 < L1 < L2 < (L3 || L4) < L5`). Every design boundary separates hardware law, multi-shape dispatch, analytical/compiler performance modeling, agent synthesis (`L3`), and NPU execution (`L4`) into orthogonal planes so no heuristic regression can mask correctness errors or mislead evolution.

Companion specifications:

- [MECE Contracts and Routing](./01_mece_contracts_and_routing.md)
- [Residual Convergence and Gates](./02_residual_convergence_and_gates.md)
- [Ascend Target Plugin README](../../targets/ascend/README.md)
- [Project Governance & Engineering Rules](../../AGENTS.md)

## Hardware Grounding

- **Execution Engines**: Vector-only (`aiv_only`), Cube-only (`aic_only`), and coupled cross-core `1C2V` (`coupled_1c2v`).
- **On-Chip Hierarchy**: `GM` -> `L2` -> `L1` -> `L0A`/`L0B`/`L0C` -> `FixPipe`, plus mode-aware `UB` (`216 / 248 / 256 KiB`) governed by the 24-bit `NodeResourceMask`.
- **Structural Cliffs**: Slowdowns stem from compiler/lowering cliffs—premature `ascend_compile_session` exit, scalar compute fallbacks, narrow unaligned DCache bypass stores, register stack spills, pseudo double buffering, and cross-regime helper clobbering.

## Five Orthogonal Architectural Planes

| Plane | Core Modules & Symbols | Orthogonal Responsibility |
| :--- | :--- | :--- |
| Plane 1: Hardware Facts, SIL & Compile Store (`L0`–`L2`) | `PyTorchSemanticAnalyzer`, `ast_semantic_gate.py`, `SILVerifier`, `ArtifactStore`, `ReadOnlyAscendPassTap`, `EnvelopeInspector` | Single semantic frontend, contiguous `ascend_compile_session` through `tilelang.lower`, positional anchor binding + non-zero divisor guard (`PrimFunc.specialize` + `arith.Analyzer`), Two-Snapshot diff, 14 hardware redlines + C++ CFG flag balance |
| Plane 2: Multi-Shape Regime Dispatch & Gates (`L2` / `L5`) | `CasePhysicsRatios`, `ascend/schedule/regime_solver.py`, `agent/gates/{gate_funnel,stage_b_gate,cpp_slice_protocol}.py`, `PromotionCoordinator` | `_subatom` separation (`contig_atom_ratio < 1.0`), weighted lexicographic `2..4` bounded partition, Post-Selector `StageBCompileGate` quotient pre-compile, CoW `_{regime_tag}_*` isolation, $3\sigma_{\text{paired}}$ CEGAR guardrail, `Delta\|Cache\| == 0` |
| Plane 3: BiSheng Sink-Down & Latency Algebra (`L0`/`L1`/`L3`) | `CompilerBehaviorRegistry`, `HybridAscendCostModel`, `latency_algebra`, `ExperienceLake` | Deterministic sink-down across 9 detectors into additive $\mu\text{s}$ conservation + `ArtifactStore`, `TopologyAxes` bucketed Tier-2 Ridge calibration (`12`/`8` specs), `cost_model_mode="off"` default (`off`/`shadow`/`advisory`), `\|ln(m/a)\| > 2.5` quarantine |
| Plane 4: Dual-Track Synthesis Flywheel (`L2`/`L3`) | `CapabilityGate`, `ISAGapScanner`, `agent/orchestrator/{llm_constraints,portfolio_runner,regime_slice_patcher,seed_synthesizer}.py`, `seed_rules.yaml`, `FlywheelCoordinator` | Zero-phi MECE `CapabilityGate`, lossless AST/TU closure views + Mode A/B Track-B patching, `max_sandbox_tool_calls=0`, `13+1` closed `SymptomFamily` simplex -> 41 atomic KB rules (`R01..R41`) + `TypedFault` infra-immune Bandit, $3\sigma_{\text{paired}}$ flywheel |
| Plane 5: Six-Tier Verification & `AgentGates` (`L0`/`L4`/`L5`) | `scripts/*_lint.py`, `tests/ascend/l0_foundation/`, `agent/npu/{ag_transport,l2_paired_eval,l3_l4_gates,msprof_probe}.py`, `l3_contract.py`, `test_npu_acceptance.py` | Parallel `L0`/`L1` + closure `ArtifactStore` -> `L2` lowering/paired bursts -> `L3` replay idempotence, CEGAR guardrails, & `L3Resolution` deferral -> `L4` CANNBench via `AgentGates` v2 with `device_fault` (`rc=75`) isolation |

## Six-Stage End-to-End Lifecycle

| Stage | Primary Activity | Owning Planes | Core Verification Output |
| :--- | :--- | :--- | :--- |
| Stage 1 | `PyTorchSemanticAnalyzer` contract parsing, unified `ascend_compile_session`, anchor specialization, `ReadOnlyAscendPassTap` + Two-Snapshot diff | Plane 1 | Lowered IR envelope & safety diagnostics |
| Stage 2 | `CasePhysicsRatios`, `2..4` regime partition, CoW helper isolation, `RegimePiecewiseSolver` | Plane 1, Plane 2 | Regime dispatch table & zero sibling clobbering |
| Stage 3 | 9-detector sink-down (7 baseline + 2 sink-down incl. Track-B UB/GM loops & DCache bypass narrow stores), Tier-0/1 analytical penalties, Tier-2 residual scoring | Plane 3 | Ranked candidates & `unmodeled_anomalies.jsonl` quarantine |
| Stage 4 | Zero-phi MECE `CapabilityGate` routing, `cann_9_3` on-demand Layer-2 pruning (`G1..G5`), Stage-0 `+4` refinement | Plane 4 | Validated IR/source mutations preserving parent monotonicity |
| Stage 5 | `L0`–`L3` host verification: 6 lints, key audit, two-pass cache idempotence, tail guard | Plane 2, Plane 5 | Signed host pass (`tilefoundry-npu-cpu`) |
| Stage 6 | `L4` worktree-isolated CANNBench on leased NPU, 41 orthogonal `R01..R41` rules + `l2_pmu_available` guard, `FlywheelDistiller` | Plane 3, Plane 4, Plane 5 | `cannSeries`-bound signed `v2` receipt (`tilefoundry-npu`) & KB distillation |

## Eight Non-Negotiable Guardrails & Anti-Pattern Summary

1. **Contiguous Compile Session**: Keep `ascend_compile_session` active across `PyTorchSemanticAnalyzer`, anchor specialization, and `tilelang.lower` + `ReadOnlyAscendPassTap` (anti-pattern: exiting `_ascend_ctx()` before lowering).
2. **Pure Cache Keys & No Magic Caps**: Never cap `_KERNEL_CACHE` by heuristic constants (e.g., `len(cases) // 2`) or embed float-tol/host-ptr keys; enforce `L3` two-pass `Delta|Cache| == 0`.
3. **Bounded Regimes & CoW Isolation**: Separate `_subatom` from aligned reductions, merge into `2..4` regimes (`1 <= min <= len <= 4`), namespace helpers `_{regime_tag}_*`, prune dead helpers, and enforce `>= 0.97x` active / `>= 0.98x` sibling tail guards.
4. **Orthogonal 3-Layer Prompt & `G5` Lock**: Keep `L1/L2/L3` strict with zero Layer 1.5; enforce `G1..G4` + paired escapes in L1, `G5` monotonicity on syntax repair/Stage-0 retries, and `Rule-Blind` bounded pass-through (`<= 1.5x` parent analytical) in `explore_creative`.
5. **Compiler Sink-Down Over Prompt Patches**: Register structural slow paths (`scalar_compute_loop_fallback`, `dcache_bypass_narrow_store`, spills, membars) in `CompilerBehaviorRegistry` and Tier-0/1 models.
6. **Zero Op-Name Branching**: Generic SIL, `EnvelopeInspector`, CostModel, and `CapabilityGate` passes dispatch strictly on physical ratios, `TopologyAxes`, and IR attributes—never operator names.
7. **Honest Gates & Anomaly Quarantine**: Never swallow exceptions, fall back silently to default schedules, relax assertions, or fit `|ln(m/a)| > 2.5` outliers into Ridge calibration.
8. **Execution & Env Isolation**: Never export `TILEFOUNDRY_NPU_*` in parent shells or create `.venv`/`venv` inside the repo; execute host/NPU suites via `AgentGates` v2 adapters, and run inner campaign workers in clean empty temp dirs without reading repo `.md` files.

Visual system views: [E2E Architecture SVG](../reports/tilefoundry_ascend_e2e_architecture.svg) | [5-Plane System MMD](../reports/tilefoundry_ascend_architecture.mmd) | [6-Stage Flow MMD](../reports/tilefoundry_ascend_e2e_flow.mmd)
