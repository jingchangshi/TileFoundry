# TileFoundry Ascend 950PR Architecture, Core Mechanisms, and E2E Evolution Guide

## 1. Three-Layer Backend & Compilation Architecture

TileFoundry targets Ascend 950PR (`ascend.950pr` / `DaVinci950`, compiler architecture `dav_3510`) through a three-layer architecture that decouples hardware-document modeling, target-aware HIR/TIR lowering with dual-channel CPU checks (`evaluate` + `verify_prim_function`), Z3 Modulo pipeline scheduling, and production TileLang NPU Mix/`SimdVF` kernel execution:

```mermaid
flowchart TD
    L1["Layer 1: Target Hardware Modeling, 50 TIR Instructions, and Z3 Scheduler Plugin\n(targets/ascend/tilefoundry_ascend/ascend/)"]
    HW["target/hardware/{davinci_950,ascend_950pr}.toml + target/{architecture,device,spec,facts,target}.py\nFact Projection + 216/248/256 KiB Three-Tier UB Model + PURE_SIMDVF_UBUF_FLAGS"]
    IR["ir/execution.py (SimdVF/SimtVF) + ir/tir/instructions.py (50 T.ascend_* ops)\nvisitor_registry/candidates.py (Unified HIR Mapping + 16 Reshard Candidates)"]
    PASS["passes/pipeline.py (ValidateAscendPipelines, ResolveAscendPipelines)\npasses/physical_ledger.py + schedule/{pipeline_solver,envelope_inspector}.py"]

    L2["Layer 2: Authored HIR -> Scheduled Mix HIR -> Lowered TIR\n(examples/ascend950pr_ops/<op>/)"]
    MODEL["model.py: Pure Tensor Math Semantics -> evaluate() CPU Alignment"]
    SCHED["scheduled.py: Mesh(aiv=2) + Mesh(aic=1) + T.SimdVF() + tf.pipelined()"]
    TIR["tir/<op>.py: check_program() -> analyze() -> finalize() -> as_script() + verify_prim_function()"]

    L3["Layer 3: Production TileLang NPU Kernel Execution\n(examples/ascend950pr_ops/<op>/kernels.py)"]
    TLK["@T.prim_func + T.MixedKernel(28, sids=2) + T.SimdVF()/T.simd.* + T.dual_copy + PURE_SIMDVF_UBUF_FLAGS"]
    NPU["tilelang.compile(target='ascend') -> _KERNEL_CACHE (bm=128 MLA / tile_m AR) -> Ascend 950PR"]

    L1 --> HW
    L1 --> IR
    L1 --> PASS
    L2 --> MODEL
    MODEL --> SCHED
    SCHED -->|"BEFORE_LOWER -> Lowering -> AFTER_LOWER"| TIR
    PASS -.->|"Dual-Mode Z3 & Envelope Analysis"| TIR
    L2 --> L3
    L3 --> TLK
    TLK --> NPU
```

---

## 2. Layer 1 Core Mechanisms and Design

### 2.1 Fact Projection: Cross-Backend Decoupling and Least Privilege

Within `targets/ascend/tilefoundry_ascend/ascend/target/{facts.py,target.py,spec.py}`, `target.get_facts(FactType)` enforces both horizontal decoupling and vertical isolation:

1. **Decoupling Across Backends**: Core compiler components (memory `analyze`, candidate filtering `candidates`, lowering `finalize`, schedule verification) never bind to backend-private TOML or device fields across NVIDIA CUDA, Apple Metal, or Ascend NPU. Instead, they query canonical protocol facades (`TopologyFacts`, `MemoryHierarchyFacts`, `MemoryExecutionFacts`, `ThroughputFacts`, `PerformanceServiceFacts`). Hardware differences remain confined inside each target's `facts.py` projection.
2. **Least Privilege and Evidence Isolation**: Each hardware facade exposes only the minimum fields required for its decision scope (for example, `AscendSpaceFacts` exposes only on-chip capacities and alignment constraints). `check_document_evidence` rejects policy or topology override injections inside hardware TOML specifications so optimization policy never pollutes physical device tables.

### 2.2 216 / 248 / 256 KiB Mode-Aware Three-Tier AIV UB Model

Under Ascend `dav-3510` hardware rules and default CCE compiler behavior, each AIV owns 256 KiB (262,144 B) of physical Unified Buffer SRAM, yielding three distinct usable capacities by execution mode:

| Mode / Compiler Flags | Usable UB | Byte Limit | Hardware & Compiler Reservation Notes |
| :--- | ---: | ---: | :--- |
| **Hybrid / `SimtVF` Mode** | **216 KiB** | `221,184 B` | Subtracts 32 KiB SIMT Data Cache (`SHARES_CAPACITY_WITH` physical SRAM) plus 8 KiB default compiler VF stack / ASC reserve |
| **Pure `SimdVF` Default** | **248 KiB** | `253,952 B` | No SIMT memory accesses; skips 32 KiB SIMT Data Cache partition and keeps 8 KiB default VF stack / ASC reserve |
| **Pure `SimdVF` Unreserved** | **256 KiB** | `262,144 B` | Releases the full 256 KiB when `PURE_SIMDVF_UBUF_FLAGS = ("--cce-disable-vf-stack-reserved-ubuf", "--cce-disable-asc-reserved-ubuf")` is active without register spill |

- **Data Model Grounding**: `ascend_950pr.toml` declares `simdvf_usable_per_aiv = 253952`; `facts.py` models the implicit `simt_dcache` (32 KiB) `SHARES_CAPACITY_WITH` constraint inside `MemoryHierarchyFacts` and provides `AscendSpaceFacts.ub_limit_bytes(mode, unreserved_ubuf=...)`.
- **Mode-Aware Z3 Constraints**: `_contains_simtvf(fn_stmt)` in `pipeline_solver.py` detects `ExecutionScope("ascend.simtvf")`, relaxing multi-buffer capacity limits for pure `SimdVF` loops to `248 KiB` (or `256 KiB` when `unreserved_ubuf=True`) while keeping `SimtVF`-containing loops strictly at `216 KiB`.

### 2.3 50 Generalized `SimdVF` TIR Instructions and Unified HIR Lowering

`targets/ascend/tilefoundry_ascend/ascend/ir/tir/instructions.py` provides **50 `T.ascend_*` instructions** (each with `typeinfer`, ISL `access_relation`, CPU `schedule_eval`, and `verify_stmt`):

1. **Cube MMAD (1 op)**: `AscendMmad` (`aic` mesh, `f32` `L0C` accumulation).
2. **Unified Semantic HIR Target Family (5 ops)**: `AscendVbinary` (`kind: BinaryKind` covering `ADD`, `SUB`, `MUL`, `DIV`, `MAX`, `MIN`, `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`), `AscendVunary` (`kind: UnaryKind` covering `NEG`, `ABS`, `EXP`, `LOG`, `SQRT`, `RSQRT`, `SIGMOID`, `TANH`), `AscendVreduce` (`kind: ReduceKind` + `keepdim` covering `SUM`, `MAX`, `MIN`, `PROD`), `AscendVwhere`, and `AscendVclamp`.
3. **Explicit Predicate, Scalar-Vector, Index, and FMA Ops (27 ops)**:
   - Arithmetic, reduction, and cast ops: `AscendVadd`, `AscendVsub`, `AscendVmul`, `AscendVdiv`, `AscendVmax`, `AscendVexp`, `AscendVrsqrt`, `AscendVcast`, `AscendVreduceSum`, `AscendVreduceMax`, `AscendVsel`;
   - Predicate synthesis and logic: `AscendPset`, `AscendUpdateMask`, `AscendPand`, `AscendPor`, `AscendPxor`, `AscendPnot`, `AscendVcmp`, `AscendVcmps` (enforcing 32-bit `elem_width=32` `SIMDVF_KIND` predicate granularity);
   - Scalar-vector, broadcast, index, and fused multiply-add: `AscendVadds`, `AscendVmuls`, `AscendVmaxs`, `AscendVmins`, `AscendVdup`, `AscendVdupv`, `AscendVci` (`INC_ORDER`/`DEC_ORDER`), `AscendVmula`.
4. **Core & 4 Extended `Reshard` / DMA / `dual_copy` Ops (17 ops)**: Detailed in Section 2.4.

> **Why Unified `HirBinary` Avoids `ScheduleRefusal` Ambiguity**:
> - NVGPU and core `common.py` bind `HirBinary` to a single candidate `(TirBinary,)` carrying `kind = ParamDef(attribute, BinaryKind)`, so `select_instruction` sees `len(fitting) == 1`.
> - Registering five separate classes `(AscendVadd, AscendVsub, AscendVmul, AscendVdiv, AscendVmax)` without a `BinaryKind` discriminant shares identical `(RMEM, RMEM) -> RMEM` placement and hits `len(fitting) == 5`, raising `ScheduleRefusal`.
> - Ascend maps `HirBinary -> (AscendVbinary,)`, `HirUnary -> (AscendVunary,)`, `HirReduce -> (AscendVreduce,)`, `HirWhere -> (AscendVwhere,)`, and `HirClamp -> (AscendVclamp,)`, propagating `kind` through `candidate_operands` for zero-annotation HIR lowering to `SimdVF` TIR.

### 2.4 NVGPU vs Ascend `Reshard` Multi-Candidate Design and 4 Redistribution Extensions

On NVGPU (`targets/nvidia`), `Reshard` binds `(CopyAsync, CopyAsyncTensor, CopyReduceAsyncTensor, LdMatrix, StMatrix)` plus core `Copy` into a 6-candidate list (covering async gather, TMA 5D box swizzle, L2 atomic reduction writeback, and warp matrix layout instructions), requiring explicit `tf.schedule(..., op=...)` selection when `len(fitting) != 1`.

Without altering core `src/tilefoundry/`, Ascend preserves unambiguous 1-to-1 automatic defaults across storage transfer directions while registering **16 `Reshard` candidates** across **4 redistribution families** using exact `ParamDef.pattern.storage` checks and constrained predicates:

| Extension Class | Added TIR Instructions | Engine / Scope | Redistribution Capabilities |
| :--- | :--- | :--- | :--- |
| **1. Cross-Core On-Chip `dual_copy`** | `AscendDualCopyL0cToUb`, `AscendDualCopyUbToL1` | `dual_copy_engine` / `ascend.dual_copy` (`_OUTSIDE_VF`) | Symmetric M/N `tmem <-> smem` split transfer with optional ND-to-NZ conversion and automatic default `tmem<->smem` selection |
| **2. Format Conversion & Padding DMA** | `AscendGmToL1Nz`, `AscendGmToUbPad` | `mte2_engine` / `ascend.dma` (`_OUTSIDE_VF`) | Hardware `nd2nz`/`dn2nz` fractal conversion, 2D tail transpose (`transpose=True`), and `pad_value` right-padding for unaligned trailing axes |
| **3. Structured `SimdVF` Register Dist** | `AscendVldDist`, `AscendVstsDist` | `vector_engine` / `ascend.vector` (`_INSIDE_VF`) | `NORM_B32`/`US_B16`/`BRC_B32` `(M,1)->(M,N)` broadcast load and `NORM_B32`/`PK_B32`/`ONEPT_B32` `(M,N)->(M,1)` lane-0 scalar store |
| **4. Writeback Hardware Atomics** | `AscendUbToGmAtomic`, `AscendL0cToGmAtomic` | `mte3_engine` / `fixpipe_engine` (`ascend.dma`) | Marks `dst` `READ \| WRITE` for store-mode DMA in-place `add`/`max`/`min` reductions; models `L0C->GM` FixPipe `quant_pre` `f32->f16/bf16` |

### 2.5 `dav-3510` Writeback Hardware Accumulation Model (`T.copy` & `T.dual_copy`)

Across TileLang Ascend IR (`T.copy`, `T.dual_copy`, `T.set_atomic`, `RewriteDualCopy`, `CodeGenAscend`) and CANN primitives (`asc_copy_l0c2gm`, `asc_copy_ub2gm_align`, `asc_copy_l0c2ub`, `ST_L2CacheType`), hardware writeback accumulation follows three strict architectural boundaries:

1. **Stateful Store-Mode DMA Atomic Register (`SpecialRegister::kAtomicMode`)**: Tile writebacks use per-core stateful control registers configured via `T.set_atomic(op="add"|"max"|"min", dtype=...)` (`PIPE_S`) and reset via `T.set_atomic_none()`. AutoSchedule inserts `PIPE_S -> PIPE_MTE3` and `PIPE_S -> PIPE_FIX` barriers for all GM-writing `T.copy` and `T.dual_copy` ops with `ReadsAtomicRegister() = true`, enabling L2/memory-controller read-modify-write across `fp32`/`fp16`/`bf16`/`int8`/`int16`/`int32`.
2. **Two GM Atomic Writeback Paths and `l2_cache_ctrl`**:
   - **AIC FixPipe Path (`T.copy(l0c, gm)` -> `asc_copy_l0c2gm`)**: Fuses `L0C` readout, `NZ2ND` conversion, optional `quant_pre` (`f32 -> f16/bf16`), and GM atomic reduction in one hardware pass.
   - **AIV MTE3 Path (`T.copy(ub, gm)` / `T.dual_copy(ub, gm)` -> `asc_copy_ub2gm_align`)**: `RewriteDualCopy` (`kSoftwareUBToGM`) splits M/N work per `sid` across dual AIVs with L2 reduction hints (`NORMAL_RED=3`, `NOTALLOC_RED=7`, `WBH_RED=11`, `WTS_RED=15`).
3. **On-Chip `L0C -> UB` Bridge (`T.dual_copy(l0c, ub)`) Boundary**: `kHardwareL0CToUB` (`asc_copy_l0c2ub`) overwrites dual-AIV UBs on `PIPE_FIX` (`eltwise_op = DEFAULT`) and **does not** support hardware atomic accumulation or `quant_pre` (both `src` and `dst` must remain `float32`). Same-block QK accumulation stays in `L0C` (`clear_accum=False`), whereas cross-KV online-softmax rescaling lands in temporary UB slots via `T.dual_copy(o_l0c, O_tmp_ub)` for `T.SimdVF()` vector accumulation.

---

## 3. Key Ascend 950PR Performance Optimizations

1. **1-AIC : 2-AIV Coupled Mix Execution**: `T.MixedKernel(28, sids=2)` partitions `bm` evenly into `rows_per_sid = bm // 2`, halving single-AIV UB footprint; `pipeline_solver.py` models this via `aiv_factor = 2`.
2. **Pure `SimdVF` 248 / 256 KiB UB Unlocking & Expanded `TileSize`**:
   - **MLA**: BSND GQA (`n_kv == 1` folded into contiguous M) enables `bm = 128` (`rows_per_sid = 64`) and direct `GM -> L1` `aligned_group_q` loads without `UB -> L1` detours, using split `O_tmp0_ub`/`O_tmp1_ub` (`8 x d_nope`) dual writebacks and `PURE_SIMDVF_UBUF_FLAGS` to reach **0.3129 ms (109+ TFLOPS)** on `mla_11`.
   - **AttentionResidual**: Computes `_choose_ar_tile_m` against the `256 KiB` pure-`SimdVF` budget (`D=2048..2560` -> `tile_m=8`, `D=3072` -> `7`, `D=4096` -> `5`, `D=5120/6144` -> `3`, `D=7168` -> `2`) with an isolated `v_tail_ub` tail block preserving `num_stages=2` multi-buffering in the main loop.
3. **Single-Pass Online Flash Softmax with Predicate Protection**: Maintains `f32` `(m_ub, l_ub, O_ub)` in UB with `m_safe = max(m_new, -1e30)` and `vcmps`/`vsel` valid-column masking to prevent `NaN` on fully masked causal rows.
4. **Cross-Iteration (`distance=1`) Z3 Modulo Solving**: `pipeline_solver.py` detects read-before-write loop-carried state and fixes `fixed=1` while solving optimal intra-iteration stage assignment and multi-buffer versions.

---

## 4. Six-Stage MLA End-to-End (E2E) Lowering Flow

```mermaid
flowchart TD
    S1["Stage 1: examples/ascend950pr_ops/mla/model.py\nAuthored HIR (MlaModel.forward, BSND/BNSD, causal/non-causal)"]
    S2["Stage 2: examples/ascend950pr_ops/mla/scheduled.py\nScheduled MIX HIR (Mesh aiv=2 + Mesh aic=1 + T.SimdVF + tf.pipelined)"]
    S3["Stage 3: check_program -> analyze -> finalize\nValidateAscendPipelines + HIR->TIR + ResolveAscendPipelines (Z3 II=10)"]
    S4["Stage 4: examples/ascend950pr_ops/mla/tir/mla.py\nCanonical Lowered TIR (as_script, fixed_ii=10, 0 SimtVF)"]
    S5["Stage 5: examples/ascend950pr_ops/mla/kernels.py\nTileLang NPU Kernel (bm=32/64/128, aligned_group_q, PURE_SIMDVF_UBUF_FLAGS)"]
    S6["Stage 6: tests/ascend/benchmarks/test_npu_acceptance.py\n20 cannbench cases + 8 GQA/tail cases + dynamic shape cache reuse"]

    S1 -->|"evaluate() vs golden.py"| S2
    S2 -->|"candidates() + evaluate()"| S3
    S3 -->|"verify_prim_function()"| S4
    S4 -->|"Lowered schedule blueprint"| S5
    S5 -->|"tilelang.compile(target='ascend')"| S6
```

1. **Stage 1 (`model.py`)**: Expresses pure tensor math (`concat([q_nope, q_rope])`, KV-head repeat, `tf.matmul`, scaling, causal mask, `tf.softmax`, `probs @ v_h`), validated against `golden.py` via CPU `evaluate`.
2. **Stage 2 (`scheduled.py`)**: Binds hardware primitives via `candidates()` (`T.ascend_mmad`, `T.ascend_dual_copy`, `T.ascend_vcast`, `T.SimdVF()`, `tf.pipelined(kv_tiles, max_stage=2, max_buffer_versions=2, enable_offset=True)`) and verifies numerical equivalence on CPU.
3. **Stage 3 (`check_program -> analyze -> finalize`)**: Executes `ValidateAscendPipelines`, `ConvertHIRToTIR`, `ResolveAscendPipelines` (Z3 `fixed_ii=10` within 248 KiB pure `SimdVF` UB), and `verify_prim_function`.
4. **Stage 4 (`tir/mla.py`)**: Emits standardized TIR (`as_script`) carrying resolved pipeline decisions and zero `SimtVF` scopes.
5. **Stage 5 (`kernels.py`)**: Builds the `@T.prim_func` (`T.MixedKernel(28, sids=2)`, `T.dynamic`, `_choose_mla_bm`, `PURE_SIMDVF_UBUF_FLAGS`, split-`D_nope` writeback) cached in `_KERNEL_CACHE`.
6. **Stage 6 (NPU Hardware Acceptance)**: Runs 20 `cannbench` cases, 8 GQA/tail cases, and dynamic cache reuse checks on Ascend 950PR (latest latency records in [ascend950pr_adaptation.md](./ascend950pr_adaptation.md)).

---

## 5. Autonomous Causal Evolution and 5-Tier Campaign Architecture

Beyond static lowering, `targets/ascend/tilefoundry_ascend/agent/` closes the loop from L0 HIR pre-routing to verified knowledge distillation while preserving strict compiler rule-blindness (`targets/ascend/tilefoundry_ascend/ascend/` never imports `tilefoundry_ascend.agent` or operator names):

```mermaid
flowchart TD
    HIR["Authored HIR / Schedule Input"]
    CAP["L0 Pre-Routing (capability_gate.py)\nextract_structural_features() + route_execution_track() + Optional TIR Blueprint (as_script)"]
    SEED["Tier-0 Seed Synthesis (e2e_evolution.py)\nbuild_hir_seed_request() -> @T.prim_func DSL Seed"]

    PRESCREEN["Neighborhood Prescreen (tilesize_prescreen.py)\nDeterministic tile_kwargs pruning -> Static prescreen summary"]
    STAGEA["Dual-Mode Stage A & Physical Ledger\nMode A: 24-bit Z3 solver (pipeline_solver.py) for T.Pipelined II\nMode B: Two-Snapshot IR Diff (envelope_inspector.py -> physical_ledger.py)"]

    KB["BayesianRuleTracker (agent/kb/)\nTrue rule.matches(ctx) using parent.features + parent.residual"]
    EXPLOIT["Rule-Exploit Workers (up to K)\nMatched rules only; no active_rules padding;\nTransport errors exempt from negative KB trials"]
    EXPLORE["Rule-Blind Creative Workers (1 + unfilled slots)\nOrthogonal focus hints + NEGATIVE CONSTRAINTS + prescreen summary;\nZero R01..R12 rule IDs or YAML leakage"]

    L0["Tier L0: Dual-Mode Stage A + Prescreen Gate"]
    L1["Tier L1: Compile & Smoke Check"]
    L2["Tier L2: Surrogate Anchor Batch\n(Eq. 3 Score + avg roofline_eta residual)"]
    L3["Tier L3: Full Contract Verification\n(Atomic checkpoint rollback on failure)"]
    L4["Tier L4: Official AgentGates CANNBench Gate"]

    BEAM["TwoLaneBeamSearch (residual_beam.py)\nExploited + Stepping-Stone Lanes + mark_parent_selected + Checkpoint beam_state"]
    FLY["Fail-Closed Flywheel (ablation_distiller.py)\nSingle-variable AST chunking -> 7-class parameterized micro-probes (30-50 lines)\nstmt_rewrite -> None (never admitted); Hardware gate >= 1.08x + wait-ratio check"]

    HIR --> CAP --> SEED --> PRESCREEN --> STAGEA
    STAGEA --> KB
    KB -->|"rule.matches(ctx) == True"| EXPLOIT
    KB -->|"Unfilled slots convert cleanly"| EXPLORE
    STAGEA -.->|"Excluded mechanism categories"| EXPLORE

    EXPLOIT --> L0
    EXPLORE --> L0
    L0 --> L1 --> L2 --> L3 --> L4
    L2 --> BEAM
    L3 -->|"mark_parent_selected / rollback"| BEAM
    BEAM -->|"Next-round parent selection"| STAGEA
    L3 -->|"Winning vs baseline DSL delta"| FLY
    L4 -.->|"Verified hardware winner"| FLY
    FLY -->|"Pure-English 6-tuple rules only"| KB
```

### 5.1 L0 HIR Pre-Routing & Dual-Mode Stage A Accounting

- **L0 Pre-Routing (`capability_gate.py` & `e2e_evolution.py`)**: `_inspect_hir_for_seed()` inspects CPU-loadable HIR modules with `extract_structural_features` and `route_execution_track`, and exports lowered native TIR blueprints (`analyze` -> `finalize` -> `as_script`) when concrete, seeding Tier-0 `@T.prim_func` generation directly on the routed execution track (`T.MixedKernel` / `SimdVF`).
- **Dual-Mode Stage A (`envelope_inspector.py`, `physical_ledger.py`, `gate_funnel.py`)**:
  - **Mode A (Static `T.Pipelined` Modulo Solver)**: Runs 24-bit Z3 scheduling (`pipeline_solver.py`) for explicit `T.Pipelined` stages to resolve steady-state initiation interval (`steady_state_ii`) and buffer versions.
  - **Mode B (Two-Snapshot IR Diff Envelope Inspector)**: Captures `BEFORE_LOWER` and `AFTER_LOWER` lowered snapshots in `envelope_inspector.py` to quantify real on-chip waterlines, implicit compiler bloat (`peak_allocated_bytes - explicit_bytes`), and `degraded_single_buffers` without rule injection or kernel-specific heuristics, aggregated through `build_objective_physical_ledger()`.
- **Tile-Size Neighborhood Prescreening (`tilesize_prescreen.py`)**: Evaluates candidate `tile_kwargs` grids against static memory limits inside `agent/orchestrator/` before dispatching workers, passing legal neighborhoods and cliff warnings into worker prompts.

### 5.2 Portfolio Worker Allocation, Rule-Blind Constraints & 5-Tier Campaign

- **Unbiased Slot Allocation (`portfolio_runner.py`, `campaign_runner.py`)**: `plan_workers()` populates `features` and `prev` residual telemetry in `build_predicate_context`, dispatches `rule_exploit` workers strictly for matched rules (`rule.matches(ctx)`), and converts unfilled rule slots into orthogonal `creative_explore` workers with dedicated `CREATIVE_FOCUS_HINTS`. LLM transport/watchdog failures (`503`, timeouts, sandbox tool aborts) never trigger `record_outcome(..., postcondition_realized=False)`.
- **Rule-Blind Negative Constraints**: Creative workers receive generic `NEGATIVE CONSTRAINTS` mechanism summaries and prescreen reports without any `R01..R12` rule identifiers or YAML snippets.
- **5-Tier Campaign Pyramid (`agent/campaign/campaign_runner.py`)**: Candidates advance strictly through `L0 (Stage A Z3/Envelope & Prescreen) -> L1 (Compile & Smoke) -> L2 (Surrogate Anchor Batch scoring Eq. 3 + roofline_eta residual) -> L3 (Full Contract Verification with atomic state rollback) -> L4 (Official AgentGates CANNBench)`. `TwoLaneBeamSearch` persists `"beam_state"` (exploited + stepping-stone lanes and selection decay counts via `mark_parent_selected`) inside atomic checkpoints for lossless `--resume`.

### 5.3 Fail-Closed Counterfactual Micro-Probe Flywheel

- **7 Parameterized Mechanism Classes (`ablation_distiller.py`)**: `_classify_edit` and `_extract_probe_params` decompose winning AST chunks (`core_parallelism`, `coalesced_copy`, `pipeline_versions`, `ssbuf_handoff`, `barrier_prune`, `dual_copy_1c2v`, `pad_alignment`) and synthesize standalone 25-50 line isolated `(baseline_probe_source, optimized_probe_source)` pairs with concrete AST-derived dimensions, core counts, stage depths, and dtypes.
- **Fail-Closed Hardware Gate**: Unmapped arbitrary edits (`stmt_rewrite`) return `synthesize_micro_probe(chunk) -> None` and are rejected as `unverifiable_edit_kind:stmt_rewrite` with zero KB writes. Mapped probes must achieve numerical accuracy, `>= 1.08x` default hardware speedup, and non-regressing `msprof_detail` MTE2 wait ratios (`aiv_mte2`, `aiv_vec`, `aic_fixpipe`) before pure-English 6-tuple `AscendCausalRule` admission into `BayesianRuleTracker`.
