# TileFoundry Ascend Target Plugin

`tilefoundry-ascend` provides the `ascend` target backend and dual-track neuro-symbolic optimization stack for Ascend 950PR / 950DT (`DaVinci950`, `dav-3510`, `CANN 9.3` sole active line).

## Installation

```sh
python -m pip install -e . -e targets/ascend
```

## Package Architecture (`L0 < L1 < L2 < (L3 || L4) < L5`)

- **`L0` Hardware Facts, SIL & Compile Store** (`ascend.{target,sil,passes,compile_cache,templates}`, `ascend.cost_model.bisheng_res_parser`): `davinci_950.toml` + `cann_9_3.toml` (`dav-3510`, audited by `scripts/audit_tilelang_surface_sync.py`), `TileLangSemanticGate` (`ast_semantic_gate.py`), `PyTorchSemanticAnalyzer`, `SILVerifier`, `ArtifactStore`, `ObservationVector5D`, `CompilerBehaviorRegistry`, and 7 dual-track shells.
- **`L1` Specs, Conservation Algebra & KB** (`agent.{contracts,specs,kb}`, `ascend.cost_model.{experience_lake,cost_gate_policy}`): Frozen `contracts.py`, `latency_algebra.py`, `fault_types.py`, 13+1 `SymptomFamily` simplex, 41 atomic rules (`seed_rules.yaml`), and 3-axis Bayesian tracker.
- **`L2` Static Routing & Stage-A/B Gates** (`ascend.schedule`, `agent.{router,gates}`): `RegimePiecewiseSolver`, `ReadOnlyAscendPassTap` + `EnvelopeInspector` Two-Snapshot diff, zero-phi `CapabilityGate` + `ISAGapScanner`, `gate_funnel.py`, `cpp_slice_protocol.py`, and Post-Selector `StageBCompileGate` (`stage_b_gate.py`).
- **`L3` Synthesis Flywheel & Cost Model** (`agent.{orchestrator,flywheel}`, `agent.cost_advisor`, `ascend.cost_model.hybrid_cost_model`): `seed_synthesizer.py`, `portfolio_runner.py`, `llm_constraints.py`, `regime_slice_patcher.py` (`patch_regime_into_module`, `split_readonly_host_tail`), `HybridAscendCostModel` (`cost_model_mode="off"` default), and $3\sigma_{\text{paired}}$ `ablation_distiller.py`.
- **`L4` NPU Execution Plane** (`agent.{npu,probes}`): Orthogonal from `L3`; `ag_transport.py` (`AgentGates` `--source` transport), `l2_paired_eval.py` (poison-pill + paired bursts), `l3_l4_gates.py`, `msprof_probe.py`, and hardware micro-probes.
- **`L5` Campaign Loop & Tests** (`agent.campaign`, `tests/ascend/{l0_foundation,agent,benchmarks}`): `campaign_runner.py`, `promotion_coordinator.py`, `stage_gate_coordinator.py`, `marathon.py`, and mirrored `tests/ascend/l0_foundation/` + `tests/ascend/agent/l1..l5/` suites.

## Architecture Diagrams & Companion Specs

- [Ascend End-to-End Architecture SVG](../../docs/reports/tilefoundry_ascend_e2e_architecture.svg)
- [Ascend 5-Plane System Architecture](../../docs/reports/tilefoundry_ascend_architecture.mmd)
- [Ascend 6-Stage End-to-End Flow](../../docs/reports/tilefoundry_ascend_e2e_flow.mmd)
- [Architecture Overview](../../docs/ascend/00_architecture_overview.md)
- [MECE Contracts and Routing](../../docs/ascend/01_mece_contracts_and_routing.md)
- [Residual Convergence and Gates](../../docs/ascend/02_residual_convergence_and_gates.md)
- [Project Governance & Engineering Rules](../../AGENTS.md)
