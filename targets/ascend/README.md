# TileFoundry Ascend Target Plugin

`tilefoundry-ascend` provides the `ascend` target backend and dual-track neuro-symbolic optimization stack for Ascend 950PR / 950DT (`DaVinci950`, `dav_3510`).

## Installation

```sh
python -m pip install -e . -e targets/ascend
```

## Package Architecture

- `tilefoundry_ascend.ascend.target`: Document-backed `DaVinci950` (`dav_3510`) architecture, `Ascend950PR` / `Ascend950DT` device models, 24-bit `NodeResourceMask`, mode-aware UB capacities (`216 / 248 / 256 KiB`), legal DMA topology, and segmented HBM burst-bandwidth curves.
- `tilefoundry_ascend.ascend.ir` & `visitor_registry`: `SimdVF` and `SimtVF` execution domains, 19 canonical pipe- and resource-defining `T.ascend_*` TIR primitives, and 8 core HIR-to-TIR algebraic and sharding bindings.
- `tilefoundry_ascend.ascend.sil`: Scheduling-Interface-Layer (`TopologyAxes`, 7-facet MECE `ConstraintRelaxationMask`, `OpAdvisoryExtractor`, and `SILVerifier` with 14 hardware redlines plus C++ CFG cross-core flag-balance verification).
- `tilefoundry_ascend.ascend.schedule` & `passes`: Mode-A Z3 modulo pipeline solver (`ValidateAscendPipelines` / `ResolveAscendPipelines`), `RegimePiecewiseSolver` with bidirectional live-buffer fixed-point re-solving, Two-Snapshot lowering diff `EnvelopeInspector`, `ObjectivePhysicalLedger`, and dialect-neutral `HirBlueprintPacket`.
- `tilefoundry_ascend.ascend.cost_model` & `templates`: 3-tier `HybridAscendCostModel` (Tier-0 Roofline + 6-Pipe/AVE evaluator, Tier-1 compiler behavior penalties across 7 detectors, Tier-2 Ridge residual + MSProf pipe scales + pairwise Delta), `ExperienceLake` online calibration, and 7 dual-track (`tilelang_dsl` + `ascendc_template`) coordination shells.
- `tilefoundry_ascend.agent`: 8D `CapabilityGate` router (`Router=ON/OFF` ablatable) with `ISAGapScanner`, 33 cross-operator causal rules with 3-axis factorized Bayesian tracking, unified 3-layer progressive LLM constraint synthesizer (`llm_constraints.py`), counterfactual `FlywheelDistiller`, and unattended `tilefoundry-ascend-campaign` / `tilefoundry-ascend-marathon` 5-tier (`L0-L4`) verification runners.

## Architecture Diagrams

- [Ascend 4-Plane System Architecture](../../docs/reports/tilefoundry_ascend_architecture.mmd)
- [Ascend 6-Stage End-to-End Flow](../../docs/reports/tilefoundry_ascend_e2e_flow.mmd)
