# TileFoundry Ascend Governance & Sliced Routing

## Block A: Dual Axioms & Nine Non-Negotiable Guardrails

- **Axiom 1 (Hardware & Contract First)**: SIL contracts and L3/L4 CANNBench gates define correctness; never relax contracts or rely on single-shape/host-proxy fits.
- **Axiom 2 (Compiler Truth Over Heuristics)**: Sink deterministic degradations into IR lowering, `SILVerifier`, Z3, `EnvelopeInspector`, and `CompilerBehaviorRegistry`, not prompts.
1. **Zero Op-Name Branching**: Generic passes (`SIL`, `EnvelopeInspector`, `CostModel`, `CapabilityGate`) dispatch on physical ratios, `TopologyAxes`, IR attrs—never op names.
2. **Bounded `2..4` Regimes**: `partition_case_regimes` splits `_subatom` (`contig_atom_ratio<1.0`) from aligned reductions via weighted merge (`1<=min<=len<=4`).
3. **Topology Buckets & Prescreen**: `ExperienceLake` isolates `TopologyAxes` buckets; prescreen runs `CompilerBehaviorRegistry` detectors (`|ln(m/a)|>2.5`->`unmodeled_anomalies.jsonl`).
4. **Orthogonal 3-Layer Prompt & `G5`**: Zero L1.5; L1 (`G1..G4`+escapes), L2 (`cann_9_3`+`seed_rules.yaml`), L3 (KB); `explore_creative` is `Rule-Blind` (full Stage-B/L2 pass-through, no `<=1.5x` cutoff); `syntax_repair` uses compiler slices+`G5` only.
5. **Contiguous Session & Two-Snapshot**: Hold `ascend_compile_session` across `PyTorchSemanticAnalyzer`, `PrimFunc.specialize`+`arith.Analyzer`, `tilelang.lower`+`ReadOnlyAscendPassTap`; keep Two-Snapshot diff.
6. **CoW Helper & $3\sigma$ Guards**: `patch_regime_into_module` namespaces `_{regime_tag}_*`; `PromotionCoordinator` enforces additive $\mu\text{s}$ conservation, $3\sigma_{\text{paired}}$ CEGAR non-regression, & `>=0.97x` active.
7. **Closed 13+1 Simplex & `TypedFault`**: `13+1` `SymptomFamily` (`UNMODELED_SILICON_RESIDUAL`, `seed_rules.yaml` AST whitelist), 3-axis Bayesian tracking, Scoped Tabu, & infra-fault immunity (`INFRA_*`).
8. **Pure Cache & Honest Gates**: No magic cache cap (`len(cases)//2`) or float-tol/host-ptr keys; enforce closure `ArtifactStore`, L3 `Delta|Cache|==0`, and L4 `AgentGates` v2 (`rc=75` `device_fault`).
9. **CANN 9.3 Sole Mainline**: Target `cann_9_3` (`dav-3510`) exclusively in templates, dialects, verifiers, and `audit_tilelang_surface_sync.py`.
10. **Zero-Magic CostModel (`off`)**: `cost_model_mode` (`off` default/`shadow`/`advisory`) uses paired t-stat gating; zero heuristic `eta`/latency floors; Stage-A/B legality stays on.

## Block B: Section-Sliced Read Routing (DO NOT Read Full Specs)

> **Routing Rule**: Routine subtasks **DO NOT** read [Architecture Overview](./docs/ascend/00_architecture_overview.md) and **NEVER** read full `01_*.md`/`02_*.md`; slice **ONLY** the matched `[SEC-*]` range.

| Target Code Scope | `[SEC-*]` | Exact Sliced Route |
| :--- | :--- | :--- |
| `ascend/sil/`, `pytorch_frontend.py`, `ascend/schedule/regime_solver.py` | `[SEC-C1]` | [`01_mece_contracts_and_routing.md` (`L5-L21`)](./docs/ascend/01_mece_contracts_and_routing.md#L5-L21) |
| `agent/orchestrator/{llm_constraints,portfolio_runner,regime_slice_patcher,seed_synthesizer}.py`, `campaign_runner.py` | `[SEC-C2]` | [`01_mece_contracts_and_routing.md` (`L22-L43`)](./docs/ascend/01_mece_contracts_and_routing.md#L22-L43) |
| `observation_space.py`, `agent/npu/msprof_probe.py`, `agent/kb/` | `[SEC-C3]` | [`01_mece_contracts_and_routing.md` (`L44-L70`)](./docs/ascend/01_mece_contracts_and_routing.md#L44-L70) |
| `cost_model/` | `[SEC-G1]` | [`02_residual_convergence_and_gates.md` (`L5-L32`)](./docs/ascend/02_residual_convergence_and_gates.md#L5-L32) |
| `agent/gates/`, `envelope_inspector.py`, `verifier.py` | `[SEC-G2]` | [`02_residual_convergence_and_gates.md` (`L33-L40`)](./docs/ascend/02_residual_convergence_and_gates.md#L33-L40) |
| `tests/ascend/l0_foundation/`, `l2_paired_eval.py`, `l3_contract.py`, `test_npu_acceptance.py` | `[SEC-G3]` | [`02_residual_convergence_and_gates.md` (`L41-L55`)](./docs/ascend/02_residual_convergence_and_gates.md#L41-L55) |

Index: [Ascend Plugin README](./targets/ascend/README.md) | [E2E Architecture SVG](./docs/reports/tilefoundry_ascend_e2e_architecture.svg)

## Block C: Worker Isolation, Hygiene Lints & AgentGates Execution

- **Worker Isolation**: Op-gen workers MUST run in `TemporaryDirectory` and NEVER read repo `.md` files.
- **Static Lints (`./scripts/`)**: Run `no_machine_paths_lint.py`, `spec_refs_lint.py`, `spec_rules_lint.py`, `comment_hygiene_lint.py`, `spec_entropy_lint.py`, and `forward_references_lint.py` on touched files.
- **Execution & Env**: NEVER export `TILEFOUNDRY_NPU_*` or create `.venv`/`venv` in-repo. Verify via `AgentGates` v2 (`tilefoundry-npu-cpu` / `tilefoundry-npu`).
- **Python Discipline**: Flat typed Python (no `Any`, `cast`, `# type: ignore`, `TYPE_CHECKING`), explicit failures, full assertions.
