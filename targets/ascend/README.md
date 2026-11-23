# TileFoundry Ascend Target Plugin

`tilefoundry-ascend` provides the `ascend` target backend and dual-track neuro-symbolic optimization stack for Ascend 950PR / 950DT (`DaVinci950`, `dav-3510`, `CANN 9.3` sole active line; `cann_9_2` inert placeholder).

## Installation

```sh
python -m pip install -e . -e targets/ascend
```

## Package Architecture

- `tilefoundry_ascend.ascend.target`: Unified `davinci_950.toml` + `cann_9_3.toml` (`dav-3510`, `cann_9_2.toml` inert placeholder) surface snapshots and offline compiler verifier (`sync_dav3510_cann_surface.py`), `Ascend950PR` / `Ascend950DT` device models, 24-bit `NodeResourceMask`, mode-aware UB capacities (`216 / 248 / 256 KiB`), legal DMA topology, and segmented HBM burst-bandwidth curves.
- `tilefoundry_ascend.ascend.sil`: Single semantic frontend `PyTorchSemanticAnalyzer`, Scheduling-Interface-Layer (`TopologyAxes`, 7-facet MECE `ConstraintRelaxationMask`, `OpAdvisoryExtractor`, and `SILVerifier` with 14 hardware redlines plus C++ CFG cross-core flag-balance verification).
- `tilefoundry_ascend.ascend.passes`: TileLang internal pass read-only extractor `ReadOnlyAscendPassTap`, `RegimePiecewiseSolver` with bidirectional live-buffer fixed-point re-solving, Two-Snapshot lowering diff `EnvelopeInspector`, `ObjectivePhysicalLedger`, `ObservationVector5D`, and dialect-neutral `HirBlueprintPacket`.
- `tilefoundry_ascend.ascend.cost_model` & `templates`: 3-tier `HybridAscendCostModel` (Tier-0 Roofline + 6-Pipe/AVE evaluator, Tier-1 compiler behavior penalties across 9 detectors: 7 baseline + 2 sink-down `scalar_compute_loop_fallback` covering TIR + reachable Track-B `__ubuf__`/`__gm__` loops and `dcache_bypass_narrow_store`, Tier-2 Ridge residual + MSProf pipe scales + pairwise Delta), `ExperienceLake` online calibration, and 7 dual-track (`tilelang_dsl` + `ascendc_template`) coordination shells.
- `tilefoundry_ascend.agent`: Zero-phi MECE `CapabilityGate` router (`TopologyAxes` + `ISAGapScanner` + `CasePhysicsRatios`, `Router=ON/OFF` ablatable), 13 closed `SymptomFamily` -> 41 orthogonal atomic causal rules (`R01..R41` with `l2_pmu_available` guard and zero latency-noise escapes) and 3-axis factorized Bayesian tracking, unified 3-layer progressive LLM constraint synthesizer (`llm_constraints.py`) with on-demand `cann_9_3` Layer-2 prompt pruning and Stage-0 `+4` monotonic refinement, counterfactual `FlywheelDistiller`, and unattended `tilefoundry-ascend-campaign` / `tilefoundry-ascend-marathon` 5-tier (`L0-L4`) verification runners.

## Architecture Diagrams & Companion Specs

- [Ascend End-to-End Architecture SVG](../../docs/reports/tilefoundry_ascend_e2e_architecture.svg)
- [Ascend 5-Plane System Architecture](../../docs/reports/tilefoundry_ascend_architecture.mmd)
- [Ascend 6-Stage End-to-End Flow](../../docs/reports/tilefoundry_ascend_e2e_flow.mmd)
- [Architecture Overview](../../docs/ascend/00_architecture_overview.md)
- [MECE Contracts and Routing](../../docs/ascend/01_mece_contracts_and_routing.md)
- [Residual Convergence and Gates](../../docs/ascend/02_residual_convergence_and_gates.md)
- [Project Governance & Engineering Rules](../../AGENTS.md)
