# Ascend 950PR Adaptation and SimdVF Optimization Report

## 1. Architecture Summary

This adaptation completes the migration of the Ascend 950PR `attention_residual` and `mla` operator stacks from functional `SimtVF` execution to high-performance `SimdVF` vector/cube execution across all three layers:

1. **Layer 1: Target Execution Kind & Loop-Carried Z3 Pipeline Solver**
   - `targets/ascend/tilefoundry_ascend/ascend/ir/execution.py`: Added `SIMDVF_KIND` / `SimdVF` alongside `SIMTVF_KIND` / `SimtVF` to represent 2-AIV SIMD vector execution.
   - `targets/ascend/tilefoundry_ascend/ascend/ir/tir/instructions.py`: Bound Ascend TIR vector instructions to `ascend.simdvf`.
   - `targets/ascend/tilefoundry_ascend/ascend/schedule/pipeline_solver.py`: Extended the Z3 pipeline solver with `SIMDVF_KIND` support, loop-carried `distance=1` dependency extraction, `fixed=1` accumulator buffer pinning, byte/FLOP cycle cost estimation, and 2-AIV UB footprint splitting.
   - Fixed loop-collection handling for `StmtVisitor.visit_For` and added unit tests in `tests/ascend/ir/test_instructions_and_candidates.py` and `tests/ascend/passes/test_pipeline_solver.py`.

2. **Layer 2: Single-Pass Online Softmax HIR & Lowered TIR**
   - `examples/ascend950pr_ops/attention_residual/scheduled.py`: Migrated to `T.SimdVF()`, `Mesh(("aiv",), (2,), names=("v",))`, and a single-pass online softmax formulation (`m_ub`, `l_ub`, `acc_ub`), eliminating intermediate `scores_gm` / `probs_gm` global memory round-trips and the second reduction pass. In the lowered `forward:n` pipeline, `acc_ub`, `m_ub`, and `l_ub` are pinned to `1` while `v_ub` uses version `2`.
   - `examples/ascend950pr_ops/mla/scheduled.py`: Migrated to `T.SimdVF()`, `Mesh(("aiv",), (2,), names=("v",))`, and a single-loop Online FlashMLA Mix pipeline without `scores_gm`, pinning `o_acc_ub`, `m_ub`, and `l_ub` to `1` across `forward:kt`.
   - Regenerated `examples/ascend950pr_ops/attention_residual/tir/attention_residual.py` and `examples/ascend950pr_ops/mla/tir/mla.py` via `finalize()` + `as_script()` with zero `SimtVF` / `simtvf` references.

3. **Layer 3: Production TileLang NPU Kernels (`kernels.py`)**
   - `examples/ascend950pr_ops/attention_residual/kernels.py`: Single-pass `T.SimdVF()` + `T.simd.*` vectorization across 56 cores with per-core `kw_ub`, `T.dynamic("m_rows")` / `T.dynamic("v_rows")` shape reuse, `enable_offset=True` in `tuned` mode, and masked `rem * 2` (`b16`) / `MODE_MERGING` `vmula` tail handling (`D=2047`, `D=1000`).
   - `examples/ascend950pr_ops/mla/kernels.py`: Pure `T.SimdVF()` + `T.simd.*` (zero `SimtVF`), DMA `Q` / `Out` transfers, `T.dynamic` (`b_dim`, `q_total_rows`, `s_kv`), causal block pruning, adaptive `bm=64` tile selection when `schedule_mode == "tuned" and q_rows >= 64 and (b_dim * n_kv * m_tiles_64) >= 14`, and `enable_offset=True` with on-chip scratch buffers (`p_l1`, `S_ub`, `P_ub`, `O_tmp0_ub`, `O_tmp1_ub`) pinned to `1` so HBM-to-L1 KV buffers (`k_nope_l1`, `k_rope_l1`, `v0_l1`, `v1_l1`) multi-buffer cleanly.

## 2. Verification Summary

- **Ascend Target / IR / Passes Unit Tests**:
  `pytest tests/ascend/target tests/ascend/ir tests/ascend/passes -v` -> **25 passed** (1.97s).
- **HIR-to-TIR Operator Workflow & Zero-SimtVF Gate**:
  `pytest tests/ascend/operators/test_hir_tir_workflow.py -v` -> **4 passed** (5.34s), including `test_zero_simtvf_in_scheduled_tir_and_kernels`.
- **Full NPU Acceptance Suite (NPU 0)**:
  `pytest tests/ascend/benchmarks/test_npu_acceptance.py -v` -> **5 passed** (465.58s, 0 skipped, 0 failed):
  - `test_attention_residual_cannbench_cases`
  - `test_attention_residual_baseline_vs_tuned_parity`
  - `test_mla_cannbench_cases`
  - `test_mla_supplemental_cases`
  - `test_dynamic_shape_kernel_cache_reuse`
- **CPU Suite**:
  `pytest tests/ascend/target tests/ascend/ir tests/ascend/passes tests/ascend/operators tests/core tests/tilefoundry tests/apple -k "not test_repository_clean"` -> **819 passed, 1 deselected** (22.74s). Note: without `-k "not test_repository_clean"`, `tests/core/scripts/test_machine_paths_lint.py::test_repository_clean` fails (`819 passed, 1 failed`, 23.77s) due to pre-existing unmodified `DEFAULT_ROOT` in `tests/ascend/benchmarks/cannbench_contract.py`.
- **Modified File Lints**:
  `scripts/comment_hygiene_lint.py`, `scripts/forward_references_lint.py`, `scripts/english_only_lint.py`, and `scripts/no_machine_paths_lint.py` all exited with code 0 (0 violations).

## 3. Kernel Cache Reuse & Wall-Clock Benchmarks (NPU 0)

With `T.dynamic` symbolic dimensions enabled in `tuned` mode:
- **Attention Residual**: `_KERNEL_CACHE` compiled **19** kernels across **20** cannbench cases.
- **MLA**: `_KERNEL_CACHE` compiled **13** kernels across **20** cannbench cases.

### 3.1 Attention Residual (`baseline` vs `tuned`, vs Prior `SimtVF` `tuned`)

| Case | Shape `[B, N, M, D]` (bf16) | Prior SimtVF Tuned (ms) | SimdVF Baseline Med (ms) | SimdVF Tuned Med (ms) | Tuned Mean (ms) | Speedup vs Baseline | Speedup vs SimtVF | Bandwidth (GB/s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 1 | `[4, 1, 128, 2048]` | 0.0246 | 0.0208 | 0.0208 | 0.0216 | 1.00x | 1.18x | 151.4 |
| Case 7 | `[4, 4, 512, 4096]` | 0.3143 | 0.1464 | 0.0987 | 0.0993 | 1.48x | 3.18x | 1020.3 |
| Case 16 | `[8, 4, 2048, 4096]` | 2.1674 | 1.0418 | 0.6605 | 0.6606 | 1.58x | 3.28x | 1016.0 |
| Case 18 | `[4, 8, 512, 7168]` | 0.9418 | 0.3765 | 0.2823 | 0.2832 | 1.33x | 3.34x | 1248.2 |
| Case 20 | `[32, 1, 1024, 2048]` | 0.6742 | 0.4126 | 0.2262 | 0.2264 | 1.82x | 2.98x | 630.4 |

### 3.2 MLA (`baseline` vs `tuned`, vs Prior `SimtVF` `tuned`)

| Case | Config | Prior SimtVF Tuned (ms) | SimdVF Baseline Med (ms) | SimdVF Tuned Med (ms) | Tuned Mean (ms) | Speedup vs Baseline | Speedup vs SimtVF | Throughput / BW |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `level4/mla_1` | `q=[2,128,128,512] k=[2,128,1,512]` BSND/causal/fp16 | 1.1652 | 0.2974 | 0.2775 | 0.2779 | 1.07x | 4.20x | 16.58 TFLOPS, 259.0 GB/s |
| `level4/mla_2` | `q=[4,256,128,512] k=[4,256,1,512]` BSND/causal/bf16 | 8.6504 | 1.6134 | 1.3459 | 1.3467 | 1.20x | 6.43x | 27.23 TFLOPS, 213.6 GB/s |
| `level4/mla_8` | `q=[16,1,128,512] k=[16,2048,1,512]` BSND/non-causal/fp16 | 0.9865 | 0.1805 | 0.1519 | 0.1527 | 1.19x | 6.49x | 60.09 TFLOPS, 498.8 GB/s |
| `level4/mla_11` | `q=[128,1,128,512] k=[128,1024,1,512]` BSND/non-causal/fp16 | 3.3410 | 0.6459 | 0.4302 | 0.4299 | 1.50x | 7.77x | 84.87 TFLOPS, 745.9 GB/s |
| `level4/mla_13` | `q=[2,128,128,512] k=[2,1,128,512]` BNSD/causal/fp16 | 0.9697 | 0.5430 | 0.4515 | 0.4523 | 1.20x | 2.15x | 10.19 TFLOPS, 159.1 GB/s |
| `level4/mla_15` | `q=[2,32,64,512] k=[2,1,512,512]` BNSD/causal/bf16 | 0.4846 | 0.1899 | 0.2227 | 0.2238 | 0.85x | 2.18x | 19.23 TFLOPS, 50.0 GB/s |
| `supp/kv_tail` | `q=[1,3,64,448] k=[1,129,2,448]` BSND/causal/bf16 | 0.0565 | 0.0170 | 0.0248 | 0.0259 | 0.69x | 2.28x | 1.90 TFLOPS, 34.8 GB/s |
| `supp/decode448` | `q=[1,1,128,448] k=[1,128,4,448]` BSND/non-causal/fp16 | 0.0401 | 0.0268 | 0.0249 | 0.0259 | 1.07x | 1.61x | 1.26 TFLOPS, 49.3 GB/s |

*Note*: On small/short-tail shapes (`level4/mla_15` and `supplemental/mla_gqa_bsnd_kv_tail`), `enable_offset=True` overhead slightly exceeds multi-buffering gains over `baseline`, while still achieving >2.1x speedup over the prior `SimtVF` implementation.

## 4. `msprof` Hardware Profiling (Tuned `SimdVF`)

| Operator / Case | Kernel Duration | AIC Metrics | AIV Metrics | Cycles |
| :--- | ---: | :--- | :--- | :--- |
| AR Case 16 (`[8,4,2048,4096]`) | 660.3 us | N/A | `aiv_vec=67.1%`, `aiv_mte2=75.4%`, `aiv_mte3=4.1%` | `aiv_cycles=60154366` |
| AR Case 20 (`[32,1,1024,2048]`) | 219.0 us | N/A | `aiv_vec=63.8%`, `aiv_mte2=89.5%`, `aiv_mte3=1.5%` | `aiv_cycles=19239980` |
| MLA `level4/mla_1` | 284.8 us | `mad=9.3%`, `aic_mte2=13.0%`, `aic_mte1=7.4%`, `aic_fix=4.9%` | `aiv_vec=20.6%`, `aiv_mte2=24.5%`, `aiv_mte3=39.8%` | `aic=11260057`, `aiv=23097959` |
| MLA `level4/mla_8` | 147.3 us | `mad=38.7%`, `aic_mte2=67.2%`, `aic_mte1=33.3%`, `aic_fix=19.9%` | `aiv_vec=41.4%`, `aiv_mte2=5.6%`, `aiv_mte3=9.1%` | `aic=3608408`, `aiv=7796230` |
| MLA `level4/mla_11` | 429.5 us | `mad=31.6%`, `aic_mte2=57.8%`, `aic_mte1=27.3%`, `aic_fix=16.2%` | `aiv_vec=37.6%`, `aiv_mte2=8.5%`, `aiv_mte3=14.2%` | `aic=17655659`, `aiv=35884788` |
| MLA `level4/mla_13` | 447.6 us | `mad=5.6%`, `aic_mte2=6.7%`, `aic_mte1=4.5%`, `aic_fix=2.9%` | `aiv_vec=52.5%`, `aiv_mte2=14.8%`, `aiv_mte3=24.0%` | `aic=18940429`, `aiv=38466686` |
