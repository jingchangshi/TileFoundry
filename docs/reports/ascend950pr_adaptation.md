# Ascend 950PR 适配与 SimdVF 性能优化报告

> 关于三层目标架构、核心设计决策及 MLA 端到端（E2E）编译流程的完整说明，请参阅 [ascend950pr_architecture_and_e2e.md](./ascend950pr_architecture_and_e2e.md)。

## 1. 架构演进摘要

本次改造完成了 Ascend 950PR 上 `attention_residual` 与 `mla` 算子栈从功能正确的 `SimtVF` 向高性能 `SimdVF` Vector/Cube 协同执行的三层全链路迁移：

1. **Layer 1：目标执行模式与跨迭代 Z3 流水求解器**
   - `targets/ascend/tilefoundry_ascend/ascend/ir/execution.py`：在 `SIMTVF_KIND` / `SimtVF` 之外新增 `SIMDVF_KIND` / `SimdVF`，建模双 AIV SIMD 向量执行模式。
   - `targets/ascend/tilefoundry_ascend/ascend/ir/tir/instructions.py`：将 Ascend TIR 向量指令绑定至 `ascend.simdvf`。
   - `targets/ascend/tilefoundry_ascend/ascend/schedule/pipeline_solver.py`：扩展 Z3 流水求解器以支持 `SIMDVF_KIND`、循环携带 `distance=1` 依赖抽取、`fixed=1` 累加缓冲锁定、字节/FLOP 周期估算以及双 AIV UB 占用折半计算。
   - 修复 `StmtVisitor.visit_For` 的循环收集处理逻辑，并在 `tests/ascend/ir/test_instructions_and_candidates.py` 与 `tests/ascend/passes/test_pipeline_solver.py` 补充单元测试。

2. **Layer 2：单遍 Online Softmax HIR 与 Lowered TIR**
   - `examples/ascend950pr_ops/attention_residual/scheduled.py`：迁移至 `T.SimdVF()`、`Mesh(("aiv",), (2,), names=("v",))` 与单遍 online softmax（`m_ub`、`l_ub`、`acc_ub`），消除中间 `scores_gm` / `probs_gm` 全局内存往返及第二轮归约循环。下沉后的 `forward:n` 流水线将 `acc_ub`、`m_ub`、`l_ub` 锁定为版本 `1`，`v_ub` 分配为版本 `2`。
   - `examples/ascend950pr_ops/mla/scheduled.py`：迁移至 `T.SimdVF()`、`Mesh(("aiv",), (2,), names=("v",))` 与无 `scores_gm` 的单循环 Online FlashMLA Mix 流水线，`forward:kt` 中 `o_acc_ub`、`m_ub`、`l_ub` 均锁定为版本 `1`。
   - 通过 `finalize()` + `as_script()` 重新生成 `examples/ascend950pr_ops/attention_residual/tir/attention_residual.py` 与 `examples/ascend950pr_ops/mla/tir/mla.py`，全面消除 `SimtVF` / `simtvf`。

3. **Layer 3：生产级 TileLang NPU 内核（`kernels.py`）**
   - `examples/ascend950pr_ops/attention_residual/kernels.py`：56 核并行单遍 `T.SimdVF()` + `T.simd.*` 向量化，支持单核驻留 `kw_ub`、`T.dynamic("m_rows")` / `T.dynamic("v_rows")` 缓存复用、`tuned` 模式启用 `enable_offset=True`，以及针对非对齐尾块的 `rem * 2`（`b16`）与 `MODE_MERGING` `vmula` 处理（覆盖 `D=2047`、`D=1000`）。
   - `examples/ascend950pr_ops/mla/kernels.py`：纯 `T.SimdVF()` + `T.simd.*`（零 `SimtVF`），DMA 搬运 `Q` / `Out`，采用 `T.dynamic`（`b_dim`、`q_total_rows`、`s_kv`）、因果块剪枝、在 `schedule_mode == "tuned" and q_rows >= 64 and (b_dim * n_kv * m_tiles_64) >= 14` 时自适应启用 `bm=64`，并在 `enable_offset=True` 下将片上暂存缓冲（`p_l1`、`S_ub`、`P_ub`、`O_tmp0_ub`、`O_tmp1_ub`）版本锁定为 `1`，确保 HBM-to-L1 的 KV 缓冲（`k_nope_l1`、`k_rope_l1`、`v0_l1`、`v1_l1`）稳定多缓冲。

## 2. 验证结果汇总

- **Ascend Target / IR / Passes 单元测试**：
  `pytest tests/ascend/target tests/ascend/ir tests/ascend/passes -v` -> **25 passed**（1.97s）。
- **HIR->TIR 工作流与零 `SimtVF` 门禁测试**：
  `pytest tests/ascend/operators/test_hir_tir_workflow.py -v` -> **4 passed**（5.34s），包含 `test_zero_simtvf_in_scheduled_tir_and_kernels`。
- **NPU 0 全量验收测试集**：
  `pytest tests/ascend/benchmarks/test_npu_acceptance.py -v` -> **5 passed**（465.58s，0 跳过、0 失败）：
  - `test_attention_residual_cannbench_cases`
  - `test_attention_residual_baseline_vs_tuned_parity`
  - `test_mla_cannbench_cases`
  - `test_mla_supplemental_cases`
  - `test_dynamic_shape_kernel_cache_reuse`
- **CPU 测试集**：
  `pytest tests/ascend/target tests/ascend/ir tests/ascend/passes tests/ascend/operators tests/core tests/tilefoundry tests/apple -k "not test_repository_clean"` -> **819 passed, 1 deselected**（22.74s）。注：若不加 `-k "not test_repository_clean"`，仅 `tests/core/scripts/test_machine_paths_lint.py::test_repository_clean` 因原封未动的 `tests/ascend/benchmarks/cannbench_contract.py` 中预存 `DEFAULT_ROOT` 而失败（`819 passed, 1 failed`，23.77s）。
- **代码文件静态检查**：
  `scripts/comment_hygiene_lint.py`、`scripts/forward_references_lint.py`、`scripts/english_only_lint.py` 与 `scripts/no_machine_paths_lint.py` 对全部 13 个代码/测试修改文件检查均返回 `0`（0 违规）。

## 3. 内核缓存复用与实测时延对比（NPU 0）

在 `tuned` 模式下启用 `T.dynamic` 符号维度后：
- **Attention Residual**：20 个 cannbench 用例下 `_KERNEL_CACHE` 累计编译 **19** 个内核。
- **MLA**：20 个 cannbench 用例下 `_KERNEL_CACHE` 累计编译 **13** 个内核。

### 3.1 Attention Residual（`baseline` vs `tuned`，对比原 `SimtVF` `tuned`）

| 用例 | Shape `[B, N, M, D]` (bf16) | 原 SimtVF Tuned (ms) | SimdVF Baseline 中位 (ms) | SimdVF Tuned 中位 (ms) | Tuned 均值 (ms) | 相对 Baseline 加速 | 相对 SimtVF 加速 | 有效带宽 (GB/s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 1 | `[4, 1, 128, 2048]` | 0.0246 | 0.0208 | 0.0208 | 0.0216 | 1.00x | 1.18x | 151.4 |
| Case 7 | `[4, 4, 512, 4096]` | 0.3143 | 0.1464 | 0.0987 | 0.0993 | 1.48x | 3.18x | 1020.3 |
| Case 16 | `[8, 4, 2048, 4096]` | 2.1674 | 1.0418 | 0.6605 | 0.6606 | 1.58x | 3.28x | 1016.0 |
| Case 18 | `[4, 8, 512, 7168]` | 0.9418 | 0.3765 | 0.2823 | 0.2832 | 1.33x | 3.34x | 1248.2 |
| Case 20 | `[32, 1, 1024, 2048]` | 0.6742 | 0.4126 | 0.2262 | 0.2264 | 1.82x | 2.98x | 630.4 |

### 3.2 MLA（`baseline` vs `tuned`，对比原 `SimtVF` `tuned`）

| 用例 | 配置规格 | 原 SimtVF Tuned (ms) | SimdVF Baseline 中位 (ms) | SimdVF Tuned 中位 (ms) | Tuned 均值 (ms) | 相对 Baseline 加速 | 相对 SimtVF 加速 | 算力 / 带宽 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `level4/mla_1` | `q=[2,128,128,512] k=[2,128,1,512]` BSND/causal/fp16 | 1.1652 | 0.2974 | 0.2775 | 0.2779 | 1.07x | 4.20x | 16.58 TFLOPS, 259.0 GB/s |
| `level4/mla_2` | `q=[4,256,128,512] k=[4,256,1,512]` BSND/causal/bf16 | 8.6504 | 1.6134 | 1.3459 | 1.3467 | 1.20x | 6.43x | 27.23 TFLOPS, 213.6 GB/s |
| `level4/mla_8` | `q=[16,1,128,512] k=[16,2048,1,512]` BSND/non-causal/fp16 | 0.9865 | 0.1805 | 0.1519 | 0.1527 | 1.19x | 6.49x | 60.09 TFLOPS, 498.8 GB/s |
| `level4/mla_11` | `q=[128,1,128,512] k=[128,1024,1,512]` BSND/non-causal/fp16 | 3.3410 | 0.6459 | 0.4302 | 0.4299 | 1.50x | 7.77x | 84.87 TFLOPS, 745.9 GB/s |
| `level4/mla_13` | `q=[2,128,128,512] k=[2,1,128,512]` BNSD/causal/fp16 | 0.9697 | 0.5430 | 0.4515 | 0.4523 | 1.20x | 2.15x | 10.19 TFLOPS, 159.1 GB/s |
| `level4/mla_15` | `q=[2,32,64,512] k=[2,1,512,512]` BNSD/causal/bf16 | 0.4846 | 0.1899 | 0.2227 | 0.2238 | 0.85x | 2.18x | 19.23 TFLOPS, 50.0 GB/s |
| `supp/kv_tail` | `q=[1,3,64,448] k=[1,129,2,448]` BSND/causal/bf16 | 0.0565 | 0.0170 | 0.0248 | 0.0259 | 0.69x | 2.28x | 1.90 TFLOPS, 34.8 GB/s |
| `supp/decode448` | `q=[1,1,128,448] k=[1,128,4,448]` BSND/non-causal/fp16 | 0.0401 | 0.0268 | 0.0249 | 0.0259 | 1.07x | 1.61x | 1.26 TFLOPS, 49.3 GB/s |

*注*：在 `level4/mla_15` 与 `supplemental/mla_gqa_bsnd_kv_tail` 等小规模/短尾用例上，`enable_offset=True` 的流水线同步开销略高于多缓冲收益（相对 `baseline`），但相较原 `SimtVF` 实现仍保持 2.1x 以上显著加速。

## 4. `msprof` 硬件性能画像（Tuned `SimdVF`）

| 算子 / 用例 | 内核耗时 | AIC 指标 | AIV 指标 | 周期数 (Cycles) |
| :--- | ---: | :--- | :--- | :--- |
| AR Case 16 (`[8,4,2048,4096]`) | 660.3 us | N/A | `aiv_vec=67.1%`, `aiv_mte2=75.4%`, `aiv_mte3=4.1%` | `aiv_cycles=60154366` |
| AR Case 20 (`[32,1,1024,2048]`) | 219.0 us | N/A | `aiv_vec=63.8%`, `aiv_mte2=89.5%`, `aiv_mte3=1.5%` | `aiv_cycles=19239980` |
| MLA `level4/mla_1` | 284.8 us | `mad=9.3%`, `aic_mte2=13.0%`, `aic_mte1=7.4%`, `aic_fix=4.9%` | `aiv_vec=20.6%`, `aiv_mte2=24.5%`, `aiv_mte3=39.8%` | `aic=11260057`, `aiv=23097959` |
| MLA `level4/mla_8` | 147.3 us | `mad=38.7%`, `aic_mte2=67.2%`, `aic_mte1=33.3%`, `aic_fix=19.9%` | `aiv_vec=41.4%`, `aiv_mte2=5.6%`, `aiv_mte3=9.1%` | `aic=3608408`, `aiv=7796230` |
| MLA `level4/mla_11` | 429.5 us | `mad=31.6%`, `aic_mte2=57.8%`, `aic_mte1=27.3%`, `aic_fix=16.2%` | `aiv_vec=37.6%`, `aiv_mte2=8.5%`, `aiv_mte3=14.2%` | `aic=17655659`, `aiv=35884788` |
| MLA `level4/mla_13` | 447.6 us | `mad=5.6%`, `aic_mte2=6.7%`, `aic_mte1=4.5%`, `aic_fix=2.9%` | `aiv_vec=52.5%`, `aiv_mte2=14.8%`, `aiv_mte3=24.0%` | `aic=18940429`, `aiv=38466686` |
