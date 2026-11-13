# TileFoundry Ascend targets

`tilefoundry-ascend` provides the `ascend` target (including Ascend 950PR /
DaVinci 950 architecture and device models, Ascend TIR instructions, and
Z3 modulo pipeline verification passes) for TileFoundry.

## Installation

```sh
python -m pip install -e . -e targets/ascend
```

## Package Layout

- `tilefoundry_ascend.ascend.target`: Document-backed `DaVinci950` architecture (`dav_3510`), `Ascend950PR` device model, `AscendTarget`, and hardware fact projections (`TopologyFacts`, `MemoryHierarchyFacts`, `AscendSpaceFacts`, `ThroughputFacts`, `PerformanceServiceFacts`).
- `tilefoundry_ascend.ascend.ir`: `SimdVF` and `SimtVF` execution mode declarations plus 20 registered `T.ascend_*` TIR operations with type inference, ISL access relations, CPU schedule evaluation, and statement verifiers.
- `tilefoundry_ascend.ascend.visitor_registry`: Target-scoped HIR-to-TIR candidate bindings for `MatMul`, `Cast`, and `Reshard`.
- `tilefoundry_ascend.ascend.schedule`: Z3 modulo pipeline constraint projection and solver (`pipeline_solver.py`) modeling loop-carried dependencies, 2-AIV UB splitting, and hardware cycle quanta.
- `tilefoundry_ascend.ascend.passes`: `ValidateAscendPipelines` (`BEFORE_LOWER`) and `ResolveAscendPipelines` (`AFTER_LOWER`) module passes.

## Documentation

- [Ascend 950PR Architecture and MLA E2E Guide](../../docs/reports/ascend950pr_architecture_and_e2e.md)
- [Ascend 950PR SimdVF Optimization Report](../../docs/reports/ascend950pr_adaptation.md)
