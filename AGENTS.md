# TileFoundry Ascend Governance & Sliced Routing

## Block A: Dual Axioms & Nine Non-Negotiable Guardrails

- **Axiom 1 (Hardware & Contract First)**: Correctness is defined by SIL contracts and L3/L4 CANNBench gates; never relax contracts or rely on single-shape or host-proxy fits.
- **Axiom 2 (Compiler Truth Over Heuristics)**: Sink deterministic degradations into IR lowering, `SILVerifier`, Z3 solver, `EnvelopeInspector`, and `CompilerBehaviorRegistry`, not prompt patches.
1. **Zero Op-Name Branching**: Generic passes (`SIL`, `EnvelopeInspector`, `CostModel`, `CapabilityGate`) dispatch on physical ratios, `TopologyAxes`, and IR attrs—never op names.
2. **Bounded `2..4` Regimes**: `partition_case_regimes` splits `_subatom` (`contig_atom_ratio < 1.0`) from aligned reductions via weighted lexicographic merge (`1 <= min <= len <= 4`).
3. **Topology Buckets & Prescreen**: `ExperienceLake` isolates `TopologyAxes` buckets; Tier-0/1 prescreen runs `CompilerBehaviorRegistry` detectors (`|ln(m/a)| > 2.5` -> `unmodeled_anomalies.jsonl`).
4. **Orthogonal 3-Layer Prompt & `G5`**: Zero Layer 1.5; L1 (`G1..G4` + paired escapes), L2 (`cann_9_3` TileLang/CCE + `seed_rules.yaml`), L3 (KB); `explore_creative` is `Rule-Blind` (full pass-through to Stage-B/L2, no `<= 1.5x` cutoff); `syntax_repair` uses only objective compiler slices + `G5` lock (no error-code heuristic KB).
5. **Contiguous Session & Two-Snapshot**: Hold `ascend_compile_session` across `PyTorchSemanticAnalyzer`, `PrimFunc.specialize` + `arith.Analyzer`, and `tilelang.lower` + `ReadOnlyAscendPassTap`; keep Two-Snapshot diff.
6. **CoW Helper Isolation**: `patch_regime_into_module` prefixes helpers `_{regime_tag}_*`, prunes dead helpers, and enforces `>= 0.97x` active / `>= 0.98x` sibling tail guards.
7. **Closed KB & Scoped Tabu**: Closed 13 `SymptomFamily` -> 41 orthogonal atomic rules (`seed_rules.yaml`, zero latency-noise escapes, AST whitelist), 3-axis Bayesian tracking, and `excluded_mechanisms` tabu.
8. **Pure Cache & Honest Gates**: No magic cache cap (`len(cases) // 2`) or float-tol/host-ptr keys; enforce L3 two-pass `Delta|Cache| == 0` idempotence and L4 `AgentGates` v2 receipts.
9. **CANN 9.3 Sole Active Mainline**: Target `cann_9_3` (`dav-3510`) exclusively across templates, dialect surfaces, verifiers, and audits; keep `cann_9_2` an inert placeholder (no runtime shims or header branches).
10. **Adaptive CostModel Lifecycle**: `cost_model_mode="auto"` filters degraded same-topology candidates early, then auto-disables near Roofline/launch floor (`eta >= 0.65` or headroom `<= 1.0 us`) or slow/stagnant rounds (`gain < 3%` or `stagnation >= 1`) while keeping Stage-A hard legality always on.

## Block B: Section-Sliced Read Routing (DO NOT Read Full Specs)

> **Routing Rule**: Routine subtasks **DO NOT** read [Architecture Overview](./docs/ascend/00_architecture_overview.md) and **NEVER** read full `01_*.md` or `02_*.md`; slice **ONLY** the matched `[SEC-*]` range below.

| Target Code Scope | `[SEC-*]` | Exact Sliced Route |
| :--- | :--- | :--- |
| `ascend/sil/`, `pytorch_frontend.py`, `regime_solver.py` | `[SEC-C1]` | [`01_mece_contracts_and_routing.md` (`L5-L21`)](./docs/ascend/01_mece_contracts_and_routing.md#L5-L21) |
| `orchestrator/llm_constraints.py`, `portfolio_runner.py`, `seed_synthesizer.py`, `campaign_runner.py` | `[SEC-C2]` | [`01_mece_contracts_and_routing.md` (`L22-L43`)](./docs/ascend/01_mece_contracts_and_routing.md#L22-L43) |
| `observation_space.py`, `msprof_report_parser.py`, `agent/kb/` | `[SEC-C3]` | [`01_mece_contracts_and_routing.md` (`L44-L69`)](./docs/ascend/01_mece_contracts_and_routing.md#L44-L69) |
| `cost_model/` | `[SEC-G1]` | [`02_residual_convergence_and_gates.md` (`L5-L32`)](./docs/ascend/02_residual_convergence_and_gates.md#L5-L32) |
| `gate_funnel.py`, `envelope_inspector.py`, `verifier.py` | `[SEC-G2]` | [`02_residual_convergence_and_gates.md` (`L33-L40`)](./docs/ascend/02_residual_convergence_and_gates.md#L33-L40) |
| `test_npu_acceptance.py`, `l3_contract.py`, `surrogate_cannbench.py` | `[SEC-G3]` | [`02_residual_convergence_and_gates.md` (`L41-L55`)](./docs/ascend/02_residual_convergence_and_gates.md#L41-L55) |

Index (human/plugin entry only): [Ascend Plugin README](./targets/ascend/README.md) | [E2E Architecture SVG](./docs/reports/tilefoundry_ascend_e2e_architecture.svg)

## Block C: Worker Isolation, Hygiene Lints & AgentGates Execution

- **Worker Isolation**: Inner campaign operator-gen workers MUST run in an empty temp dir (`TemporaryDirectory`) and NEVER read repo `.md` files.
- **Static Lints (`./scripts/`)**: Run `no_machine_paths_lint.py`, `spec_refs_lint.py`, `spec_rules_lint.py`, `comment_hygiene_lint.py`, `spec_entropy_lint.py`, and `forward_references_lint.py` on touched files.
- **Execution & Env**: NEVER export `TILEFOUNDRY_NPU_*` or create `.venv`/`venv` in-repo. Verify via `AgentGates` `agent-gates` signed v2 (`tilefoundry-npu-cpu` / `tilefoundry-npu`).
- **Python Discipline**: Flat typed Python (no `Any`, `cast`, `# type: ignore`, or `TYPE_CHECKING`), explicit failure semantics, and full assertion strength.
