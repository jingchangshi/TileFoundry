# TileFoundry Ascend Governance & Sliced Routing

## Block A: Dual Axioms & Eight Non-Negotiable Guardrails

- **Axiom 1 (Hardware & Contract First)**: Correctness is defined by SIL contracts, L3 replay/invariant gates, and L4 CANNBench; never relax contracts or rely on single-shape or host-proxy fits.
- **Axiom 2 (Compiler Truth Over Heuristics)**: Sink deterministic degradations into IR lowering, `SILVerifier`, Z3 pipeline solver, `EnvelopeInspector`, and `CompilerBehaviorRegistry`, not prompt patches.
1. **Zero Op-Name Branching**: Generic passes (HIR, Z3, TIR, CostModel, Router) dispatch on physical ratios, `TopologyAxes`, and IR attrs—never operator names.
2. **Bounded `2..4` Regimes**: `partition_case_regimes` splits `_subatom` (`contig_atom_ratio < 1.0`) from aligned reductions via weighted lexicographic merge (`1 <= min <= len <= 4`).
3. **Topology Buckets & Prescreen**: `ExperienceLake` isolates `TopologyAxes` buckets; Tier-0/1 prescreen runs all `CompilerBehaviorRegistry` detectors (`|ln(m/a)| > 2.5` to `unmodeled_anomalies.jsonl`).
4. **Orthogonal 3-Layer Prompt & `G5`**: Zero Layer 1.5; L1 (`G1..G4` + paired escapes), L2 (`T.ascend_*` + `seed_rules.yaml`), L3 (KB); `explore_creative` is `Rule-Blind` (`<= 1.5x` parent analytical); `syntax_repair` enforces `G5` lock.
5. **Contiguous Session & Two-Snapshot**: Hold `ascend_compile_session` across HIR-to-TIR, `PrimFunc.specialize` + `arith.Analyzer`, and `tilelang.lower`; keep `EnvelopeInspector` Two-Snapshot diff.
6. **CoW Helper Isolation**: `patch_regime_into_module` namespaces helpers `_{regime_tag}_*`, prunes dead helpers, and enforces `>= 0.97x` active / `>= 0.98x` sibling tail guards.
7. **Closed KB & Scoped Tabu**: Use closed 13 `SymptomFamily`, read-only online `seed_rules.yaml` (`anti_precondition`, AST whitelist), 3-axis Bayesian tracking, and Scoped Tabu (`excluded_mechanisms`).
8. **Pure Cache & Honest Gates**: No magic cache cap (`len(cases) // 2`) or float-tol/host-ptr keys; enforce L3 two-pass `Delta|Cache| == 0` idempotence and L4 `AgentGates` v2 receipts.

## Block B: Section-Sliced Read Routing (DO NOT Read Full Specs)

> **Routing Rule**: Routine coding subtasks **DO NOT** read [Architecture Overview](./docs/ascend/00_architecture_overview.md) and **NEVER** read full `01_*.md` or `02_*.md`; slice **ONLY** the matched `[SEC-*]` `L<start>-L<end>` range below.

| Target Code Scope | `[SEC-*]` | Exact Sliced Route |
| :--- | :--- | :--- |
| `ascend/sil/`, `pytorch_frontend.py`, `regime_solver.py` | `[SEC-M1]` | [`01_mece_contracts_and_routing.md` (`L5-L21`)](./docs/ascend/01_mece_contracts_and_routing.md#L5-L21) |
| `orchestrator/llm_constraints.py`, `portfolio_runner.py`, `seed_synthesizer.py`, `campaign_runner.py` | `[SEC-M2]` | [`01_mece_contracts_and_routing.md` (`L22-L43`)](./docs/ascend/01_mece_contracts_and_routing.md#L22-L43) |
| `observation_space.py`, `msprof_report_parser.py`, `agent/kb/` | `[SEC-M3]` | [`01_mece_contracts_and_routing.md` (`L44-L69`)](./docs/ascend/01_mece_contracts_and_routing.md#L44-L69) |
| `cost_model/` | `[SEC-G1]` | [`02_residual_convergence_and_gates.md` (`L5-L31`)](./docs/ascend/02_residual_convergence_and_gates.md#L5-L31) |
| `gate_funnel.py`, `envelope_inspector.py`, `verifier.py` | `[SEC-G2]` | [`02_residual_convergence_and_gates.md` (`L32-L38`)](./docs/ascend/02_residual_convergence_and_gates.md#L32-L38) |
| `test_npu_acceptance.py`, `l3_contract.py`, `surrogate_cannbench.py` | `[SEC-G3]` | [`02_residual_convergence_and_gates.md` (`L39-L53`)](./docs/ascend/02_residual_convergence_and_gates.md#L39-L53) |

Index (human/plugin entry only): [Ascend Plugin README](./targets/ascend/README.md) | [E2E Architecture SVG](./docs/reports/tilefoundry_ascend_e2e_architecture.svg)

## Block C: Worker Isolation, Hygiene Lints & AgentGates Execution

- **Inner Worker Isolation**: Inner marathon/campaign operator-generation workers MUST run in an empty temp dir (`TemporaryDirectory`) and NEVER mount or read repo `.md` files.
- **Static Lints (`./scripts/`)**: Run `no_machine_paths_lint.py`, `spec_refs_lint.py`, `spec_rules_lint.py`, `comment_hygiene_lint.py`, `spec_entropy_lint.py`, and `forward_references_lint.py` on touched files.
- **Execution & Env**: NEVER export `TILEFOUNDRY_NPU_*` in parent shells or create `.venv`/`venv` inside the repo. Run host/NPU verification via `AgentGates` `agent-gates` signed v2 (`tilefoundry-npu-cpu` / `tilefoundry-npu`).
- **Python Discipline**: Keep flat typed Python (no `Any`, `cast`, `# type: ignore`, or `TYPE_CHECKING`), explicit failure semantics, and full assertion strength.
