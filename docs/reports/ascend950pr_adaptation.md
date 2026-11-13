# Ascend 950PR 适配与 SimdVF 性能优化报告

> 关于三层目标架构、Fact 投影解耦、50 条广义向量与 4 类 16 候选 `Reshard` 扩展、`dav-3510` 写回累加模型及 MLA 端到端（E2E）编译流程的完整说明，请参阅 [ascend950pr_architecture_and_e2e.md](./ascend950pr_architecture_and_e2e.md)。

## 1. 架构与能力演进摘要

本次改造围绕 Ascend 950PR（`dav-3510`）完成了三层目标建模、向量与重分布 ISA 扩展，以及 `attention_residual`、`mla` 基于纯 `SimdVF` 大 UB 预算的核内分块优化：

1. **Layer 1：模式感知 3 档 UB 模型、50 条向量 ISA 与 4 类 `Reshard` 扩展**
   - **执行模式与 Fact 投影**：`targets/ascend/tilefoundry_ascend/ascend/ir/execution.py` 提供 `SIMTVF_KIND` / `SimtVF` 与 `SIMDVF_KIND` / `SimdVF`；`targets/ascend/tilefoundry_ascend/ascend/target/facts.py` 通过 `AscendSpaceFacts.ub_limit_bytes`、`PURE_SIMDVF_UBUF_FLAGS`（`--cce-disable-vf-stack-reserved-ubuf`、`--cce-disable-asc-reserved-ubuf`）与 `MemoryHierarchyFacts`（`simt_dcache` `SHARES_CAPACITY_WITH`）精确刻画 3 档可用 UB 容量：`SimtVF` 混合模式为 `216 KiB`，默认纯 `SimdVF` 为 `248 KiB`（`simdvf_usable_per_aiv = 253952`），无栈保留纯 `SimdVF` 可达 `256 KiB`。
   - **50 条广义 `SimdVF` TIR 指令与统一 HIR 下沉**：`targets/ascend/tilefoundry_ascend/ascend/ir/tir/instructions.py` 将 `ASCEND_OP_SPECS` 扩充至 **50** 条指令，覆盖统一 HIR 映射（`AscendVbinary`、`AscendVunary`、`AscendVreduce`、`AscendVwhere`、`AscendVclamp`）、谓词/掩码（`AscendPset`、`AscendUpdateMask`、`AscendPand`/`AscendPor`/`AscendPxor`/`AscendPnot`）、向量比较（`AscendVcmp`/`AscendVcmps`）及标量-向量/FMA 指令（`AscendVadds`/`AscendVmuls`/`AscendVmaxs`/`AscendVmins`/`AscendVdup`/`AscendVdupv`/`AscendVci`/`AscendVmula`）。
   - **4 类 16 候选 `Reshard` 重分布扩展**：`targets/ascend/tilefoundry_ascend/ascend/visitor_registry/candidates.py` 将 `Reshard` 扩展至 16 个目标候选（含 8 条基础 DMA/Reg 搬运与 8 条新增重分布指令：`AscendDualCopyL0cToUb`/`AscendDualCopyUbToL1`、`AscendGmToL1Nz`/`AscendGmToUbPad`、`AscendVldDist`/`AscendVstsDist`、`AscendUbToGmAtomic`/`AscendL0cToGmAtomic`），通过 `_cand_storage()` 保持单源-单目的默认选择无歧义，同时允许 `PatternMatcher` 验证并显式下沉全部 4 类能力。
   - **模式感知 Z3 流水求解器**：`targets/ascend/tilefoundry_ascend/ascend/schedule/pipeline_solver.py` 增加 `_contains_simtvf` 与 `unreserved_ubuf` 自动判断模式感知 UB 上限，并将新增 DMA/`dual_copy` 任务纳入 `_estimate_task_latency` 与 `l0c` 角色推断。

2. **Layer 2：单遍 Online Softmax HIR 与 Lowered TIR**
   - `examples/ascend950pr_ops/attention_residual/scheduled.py`：迁移至 `T.SimdVF()`、`Mesh(("aiv",), (2,), names=("v",))` 与单遍 online softmax（`m_ub`、`l_ub`、`acc_ub`），消除中间 `scores_gm` / `probs_gm` 全局内存往返。
   - `examples/ascend950pr_ops/mla/scheduled.py`：迁移至 `T.SimdVF()`、`Mesh(("aiv",), (2,), names=("v",))` 与无 `scores_gm` 的单循环 Online FlashMLA Mix 流水线。
   - 对应的 `examples/ascend950pr_ops/attention_residual/tir/attention_residual.py` 与 `examples/ascend950pr_ops/mla/tir/mla.py` 全面消除 `SimtVF` / `simtvf`。

3. **Layer 3：生产级 TileLang NPU 内核优化（`kernels.py`）**
   - `examples/ascend950pr_ops/attention_residual/kernels.py`：56 核并行单遍 `T.SimdVF()` + `@T.macro` 辅助片段（`_simd_init_row`、`_simd_step_row`、`_simd_blend_row`）；基于 `256 KiB` 纯 `SimdVF` UB 预算实现多行分块 `_choose_ar_tile_m`（公式 `(ub_budget - 12 * d_alloc) // (10 * d_alloc + 64)`，最高至 `tile_m=8`），并以独立 `v_tail_ub` 隔离尾块写回，满足 `PrepareMultiBuffer` 单一写者要求。
   - `examples/ascend950pr_ops/mla/kernels.py`：纯 `T.SimdVF()` + `T.simd.*` 向量化，支持 `o1_l0c` 与 `O_tmp1_ub`（处理 `d1 > 0 and d1 != d0` 如 `d_nope=448, d1=192` 的分片形状）；`tuned` 模式下 `_choose_mla_bm` 在满足 `input_layout == "BSND"`、`n_kv == 1`、`(d_nope <= 256 or d_nope == 512)`、`q_rows % 128 == 0` 且总块数不小于 14 时启用 `bm = 128`（`rows_per_sid = 64`），并在 `d_nope = 448` 等超预算场景自动回退 `bm = 64` 以守住 `256 KiB` 物理 UB 上限。

## 2. 最新验证结果汇总（最终代码与文档修改后）

- **Ascend 目标 / IR / Pass / 工作流单元测试**：
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest tests/ascend/target tests/ascend/ir tests/ascend/passes tests/ascend/operators -q` -> **33 passed**（11.12s）。
- **CPU + Core 全量回归测试**：
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest tests/ascend/target tests/ascend/ir tests/ascend/passes tests/ascend/operators tests/core tests/tilefoundry tests/apple -k "not test_repository_clean" -q` -> **823 passed, 1 deselected**（22.72s）。注：未过滤时 `tests/core/scripts/test_machine_paths_lint.py::test_repository_clean` 仅因原封未动的 `tests/ascend/benchmarks/cannbench_contract.py` 中预存 `DEFAULT_ROOT` 失败。
- **NPU 2 全量验收测试集（5/5 全部通过）**：
  `tests/ascend/benchmarks/test_npu_acceptance.py` -> **5 passed**（203.76s，0 跳过、0 失败）：
  - `test_attention_residual_cannbench_cases`
  - `test_attention_residual_baseline_vs_tuned_parity`
  - `test_mla_cannbench_cases`
  - `test_mla_supplemental_cases`
  - `test_dynamic_shape_kernel_cache_reuse`
- **仓库静态规范检查**：
  `scripts/comment_hygiene_lint.py`、`scripts/forward_references_lint.py`、`scripts/english_only_lint.py` 与 `scripts/no_machine_paths_lint.py` 全部通过（0 违规），且 `src/tilefoundry/` 改动为 `0` 行。

## 3. 实测时延对比（NPU 2，`SimdVF Baseline` vs `SimdVF Tuned`）

### 3.1 Attention Residual（基于 `256 KiB` 纯 `SimdVF` 多行 `tile_m` 调优）

| 用例 | Shape `[B, N, M, D]` | `SimdVF Baseline` 中位 (ms) | `SimdVF Tuned` 中位 (ms) | `Tuned` 均值 (ms) | 相对 `Baseline` 加速 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Case 1 | `[4, 1, 128, 2048]` | 0.0213 | 0.0211 | 0.0229 | 1.01x |
| Case 7 | `[4, 4, 512, 4096]` | 0.1523 | 0.0803 | 0.0805 | 1.90x |
| Case 16 | `[8, 4, 2048, 4096]` | 1.0357 | 0.4754 | 0.4754 | 2.18x |
| Case 18 | `[4, 8, 512, 7168]` | 0.3784 | 0.2572 | 0.2572 | 1.47x |
| Case 20 | `[32, 1, 1024, 2048]` | 0.4123 | 0.1483 | 0.1485 | 2.78x |

### 3.2 MLA（`bm = 128` 分块与 `d_nope = 448` `o1_l0c` 自适应 `bm = 64`）

| 用例 | 配置规格 | `SimdVF Baseline` 中位 (ms) | `SimdVF Tuned` 中位 (ms) | `Tuned` 均值 (ms) | 相对 `Baseline` 加速 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `level4/mla_1` | `[2, 128, 128, 512]` BSND causal | 0.2584 | 0.1033 | 0.1032 | 2.50x |
| `level4/mla_2` | `[4, 256, 128, 512]` BSND causal | 1.5749 | 0.6143 | 0.6145 | 2.56x |
| `level4/mla_3` | `[2, 128, 64, 448]` BSND causal | 0.1363 | 0.0521 | 0.0526 | 2.62x |
| `level4/mla_6` | `[2, 256, 128, 448]` BSND non-causal | 0.6729 | 0.3146 | 0.3147 | 2.14x |
| `level4/mla_8` | `[16, 1, 128, 512]` BSND non-causal | 0.1860 | 0.1174 | 0.1176 | 1.58x |
| `level4/mla_11` | `[128, 1, 128, 512]` BSND non-causal | 0.6585 | 0.3058 | 0.3057 | 2.15x |
| `level4/mla_13` | `[2, 128, 128, 512]` BNSD causal | 0.4874 | 0.2789 | 0.2794 | 1.75x |
| `level4/mla_15` | `[2, 32, 64, 512]` BNSD causal | 0.1936 | 0.2002 | 0.1995 | 0.97x |

*注*：历史重构前版本（NPU 0）中，原 `SimtVF` `Tuned` 在 `AR 16/18/20` 上耗时为 `2.1674 / 0.9418 / 0.6742 ms`，在 `MLA 1/2/8/11/13` 上耗时为 `1.1652 / 8.6504 / 0.9865 / 3.3410 / 0.9697 ms`。本轮扩大 `tile_m` 与 `bm = 128` 后，NPU 2 实测 `AR 16` 进一步降至 `0.4754 ms`、`MLA 1` 降至 `0.1033 ms`、`MLA 11` 降至 `0.3058 ms`。
