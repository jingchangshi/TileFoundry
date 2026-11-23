> Audience: Human / First-Time Architect Overview. Coding subtasks DO NOT read this file; follow section-sliced routes in AGENTS.md.

# TileFoundry Ascend End-to-End Architecture Overview

TileFoundry Ascend couples deterministic symbolic compilation with dual-track neuro-symbolic schedule evolution for DaVinci950 (`dav_3510`, `Ascend950PR` / `Ascend950DT`). Every design boundary separates hardware law, multi-shape dispatch, analytical/compiler performance modeling, agent synthesis, and signed verification into orthogonal planes so no heuristic regression can mask correctness errors or mislead evolution.

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
| Plane 1: Symbolic Contract & Compilation | `ascend_compile_session`, `EnvelopeInspector`, `SILVerifier`, Z3 pipeline solver | Contiguous target context through `tilelang.lower`, pre-lowering anchor specialization (`PrimFunc.specialize` + `arith.Analyzer`), Two-Snapshot diff, 14 hardware redlines + C++ CFG flag balance |
| Plane 2: Multi-Shape Regime Dispatch & Isolation | `CasePhysicsRatios`, `partition_case_regimes`, `patch_regime_into_module` | `_subatom` separation (`contig_atom_ratio < 1.0`), weighted lexicographic `2..4` bounded partition, CoW `_{regime_tag}_*` helper isolation, dead-helper pruning, dual-layer tail guard, `Delta\|Cache\| == 0` |
| Plane 3: BiSheng Sink-Down & 3-Tier Residual Cost | `CompilerBehaviorRegistry`, `HybridAscendCostModel`, `ResidualBasisRegistry`, `ExperienceLake` | Deterministic sink-down of slow patterns into Tier-0/1 penalties, `TopologyAxes` bucketed Tier-2 Ridge calibration (`12`/`8` specs), `\|ln(m/a)\| > 2.5` anomaly quarantine |
| Plane 4: Dual-Track Neuro-Symbolic Evolution | `CapabilityGate`, `ISAGapScanner`, `llm_constraints.py`, `seed_rules.yaml`, `FlywheelDistiller` | `Router=ON/OFF` dual-track (`tilelang_dsl` / `ascendc_template`), 3-layer progressive constraints (`G1..G5`), `Rule-Blind` bounded `explore_creative`, 13 `SymptomFamily` + 40 frozen KB rules + Scoped Tabu |
| Plane 5: Five-Tier (`L0`–`L4`) Verification & `AgentGates` | `scripts/*_lint.py`, `gate_funnel.py`, `l3_contract.py`, `test_npu_acceptance.py` | `L0` static/key audit -> `L1` SIL/`G1..G5` -> `L2` lowering/cost -> `L3` replay idempotence & tail guard -> `L4` CANNBench via `AgentGates` signed v2 (`tilefoundry-npu` / `tilefoundry-npu-cpu`) |

## Six-Stage End-to-End Lifecycle

| Stage | Primary Activity | Owning Planes | Core Verification Output |
| :--- | :--- | :--- | :--- |
| Stage 1 | Contract parsing, unified `ascend_compile_session`, anchor specialization, Two-Snapshot diff | Plane 1 | Lowered IR envelope & safety diagnostics |
| Stage 2 | `CasePhysicsRatios`, `2..4` regime partition, CoW helper isolation, Z3 modulo pipeline | Plane 1, Plane 2 | Regime dispatch table & zero sibling clobbering |
| Stage 3 | BiSheng pattern detection, Tier-0/1 analytical penalties, Tier-2 residual scoring | Plane 3 | Ranked candidates & `unmodeled_anomalies.jsonl` quarantine |
| Stage 4 | `CapabilityGate` routing, 3-layer `G1..G5` constraints, bounded `explore_creative` | Plane 4 | Validated IR/source mutations preserving parent monotonicity |
| Stage 5 | `L0`–`L3` host verification: 6 lints, key audit, two-pass cache idempotence, tail guard | Plane 2, Plane 5 | Signed host pass (`tilefoundry-npu-cpu`) |
| Stage 6 | `L4` official CANNBench execution on leased NPU, Bayesian update, `FlywheelDistiller` | Plane 3, Plane 4, Plane 5 | Signed NPU receipt (`tilefoundry-npu`) & KB distillation |

## Eight Non-Negotiable Guardrails & Anti-Pattern Summary

1. **Contiguous Compile Session**: Keep `ascend_compile_session` active across HIR-to-TIR, anchor specialization, and `tilelang.lower` (anti-pattern: exiting `_ascend_ctx()` before lowering).
2. **Pure Cache Keys & No Magic Caps**: Never cap `_KERNEL_CACHE` by heuristic constants (e.g., `len(cases) // 2`) or embed float-tol/host-ptr keys; enforce `L3` two-pass `Delta|Cache| == 0`.
3. **Bounded Regimes & CoW Isolation**: Separate `_subatom` from aligned reductions, merge into `2..4` regimes (`1 <= min <= len <= 4`), namespace helpers `_{regime_tag}_*`, prune dead helpers, and enforce `>= 0.97x` active / `>= 0.98x` sibling tail guards.
4. **Orthogonal 3-Layer Prompt & `G5` Lock**: Keep `L1/L2/L3` strict with zero Layer 1.5; enforce `G1..G4` + paired escapes in L1, `G5` monotonicity on syntax repair/Stage-0 retries, and `Rule-Blind` bounded pass-through (`<= 1.5x` parent analytical) in `explore_creative`.
5. **Compiler Sink-Down Over Prompt Patches**: Register structural slow paths (`scalar_compute_loop_fallback`, `dcache_bypass_narrow_store`, spills, membars) in `CompilerBehaviorRegistry` and Tier-0/1 models.
6. **Zero Op-Name Branching**: Generic HIR, Z3, TIR, CostModel, and Router passes dispatch strictly on physical ratios, `TopologyAxes`, and IR attributes—never operator names.
7. **Honest Gates & Anomaly Quarantine**: Never swallow exceptions, fall back silently to default schedules, relax assertions, or fit `|ln(m/a)| > 2.5` outliers into Ridge calibration.
8. **Execution & Env Isolation**: Never export `TILEFOUNDRY_NPU_*` in parent shells or create `.venv`/`venv` inside the repo; execute host/NPU suites via `AgentGates` v2 adapters, and run inner campaign workers in clean empty temp dirs without reading repo `.md` files.

Visual system views: [E2E Architecture SVG](../reports/tilefoundry_ascend_e2e_architecture.svg) | [5-Plane System MMD](../reports/tilefoundry_ascend_architecture.mmd) | [6-Stage Flow MMD](../reports/tilefoundry_ascend_e2e_flow.mmd)
