# Ascend 950PR TileFoundry & TileLang Operator Adaptation Report

## 1. Architecture & Target Separation

The Ascend 950PR (`ascend.950pr`, `ascend.davinci_950` / `dav-3510`) adaptation keeps target infrastructure strictly separated from operator implementations:

- **Target Plugin (`targets/ascend/`)**: Provides `AscendTarget("ascend.950pr")` and hardware capability definitions in `targets/ascend/tilefoundry_ascend/ascend/target/hardware/ascend_950pr.toml` (`block_count = 28`, `aic_count = 28`, `aiv_count = 56`, `128 GiB` HBM, `512 KiB` L1 per AIC, `64 KiB` L0A/L0B per AIC, `256 KiB` L0C per AIC, `256 KiB` nominal / `216 KiB` usable UB per AIV).
- **Trunk & CLI Support (`src/`)**: Adds `StorageKind.TMEM` support in the core IR and registers Ascend target resolution and lowered workflows via `tilefoundry.cli`.
- **Operator Workspaces (`examples/ascend950pr_ops/`)**:
  - `examples/ascend950pr_ops/attention_residual/`: Authored HIR (`model.py`), scheduled HIR (`scheduled.py`), lowered TIR (`tir/`), and native TileLang Ascend Vector kernel (`kernels.py`).
  - `examples/ascend950pr_ops/mla/`: Authored HIR (`model.py`), scheduled HIR (`scheduled.py`), lowered TIR (`tir/`), and native TileLang Ascend MIX kernel (`kernels.py`).

## 2. Hardware Adaptation & Correctness Engineering

### 2.1 AttentionResidual (`examples/ascend950pr_ops/attention_residual/kernels.py`)

- Pure Vector `T.Kernel` scaled across up to `56` AIV cores (`num_cores = min(m_rows, 56)` matching `aiv_count = 56`).
- Uses `SimtVF` to perform FP32 RMSNorm + depthwise softmax reduction and weighted residual aggregation, with configurable `schedule_mode` (`"baseline"` vs `"tuned"`).

### 2.2 Multi-Head Latent Attention (`examples/ascend950pr_ops/mla/kernels.py`)

- **MIX Execution Grid**: `T.MixedKernel(num_cores, sids=2)` mapping `28` AIC cores (`num_cores = min(total_m_tiles, 28)`) paired with 2 AIV sub-blocks (`bm = 32`, `rows_per_sid = 16`, `bn = 64`), supporting arbitrary GQA group ratios (`N_q / N_kv`), tail query/KV dimensions, and both `BSND` and `BNSD` layouts in a single logical kernel launch.
- **Unified 4D KV DMA**: `K_nope`, `K_rope`, and `V` use 4D slice `T.copy` into L1 (`nd2nz` for keys, `dn2nz` via `transpose=True` for values), leveraging hardware out-of-bounds zero fill on tail KV blocks without scalar fallback loops.
- **Per-Row Softmax `SimtVF(threads=64)`**: TileLang's `AscendAllReduce` uses a 64-thread scratch layout at fixed row offset `0`. Executing softmax inside `SimtVF(threads=64)` with `for r in T.serial(rows_per_sid):` matches the 64-thread reducer contract and eliminates cross-row UB clobber across rows `1..3 mod 4`.
- **`d_nope <= 256` Split `P @ V` Tiling**: For `d_nope = 448` or `512`, splitting `V` and `P @ V` along columns into `d0 = min(d_nope, 256)` and `d1 = d_nope - d0` ensures every `P @ V` `ascend_gemm_l1` call uses a single L0 sub-K step (`SUB_K = 1`, event `0`), preventing the next KV iteration's `Q_nope @ K_nope^T` (`sk = 0` on event `0`) from overwriting `L0A` slots still held by `PIPE_M` for `sk = 1`.
- **Tuned Schedule (`schedule_mode="tuned"`)**: Enables 2-stage KV pipelining (`num_stages = 2` when `max_kv_blocks >= 2`) and causal block-bound pruning (`use_causal_bound = True`).

## 3. Numerical & Regression Verification Summary

All NPU acceptance checks ran on host NPU 0 (`ASCEND_RT_VISIBLE_DEVICES=0`) using `tests/ascend/benchmarks/test_npu_acceptance.py`:

| Suite | Coverage | Mode(s) | Result |
| :--- | :--- | :--- | :--- |
| AttentionResidual CANNBench | 20 / 20 cases | `tuned` | Passed (`Harness.compare`) |
| AttentionResidual Baseline Parity | Cases 1, 16, 20 | `baseline` | Passed (`1` kernel launch) |
| MLA CANNBench | 20 / 20 cases | `tuned` | Passed (`Harness.compare`) |
| MLA Supplemental GQA/Tail/Layout/Scale | 4 / 4 cases | `tuned`, `baseline` | Passed (`1` kernel launch each) |

Full suite run of `tests/ascend/benchmarks/test_npu_acceptance.py`: **4 passed in 458.24s** (`0` failures, `0` skips); post-editable-install NPU 0 parity smoke run (`-k baseline_vs_tuned_parity`): **1 passed in 31.04s**.

CPU compiler/IR/CLI/lint verification (no CANN runtime loaded, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`):

- `tests/ascend` (excluding `test_npu_acceptance.py`), `tests/core`, `tests/tilefoundry`, and `tests/apple`: **817 passed in 18.96s** (`0` failures, `0` skips).
- `tests/nvidia` (excluding `tests/nvidia/cuda/integration/models`): **690 passed, 43 skipped, 18 failed** (`17` failures require a local CUDA GPU; `1` failure is a pre-existing upstream `'carrier'` storage diagnostic regex mismatch in `tests/nvidia/cuda/parser/test_functions.py`).
- Repository lints (`comment_hygiene_lint.py`, `forward_references_lint.py`, `english_only_lint.py`, `no_machine_paths_lint.py`) on all new/modified files: **0 violations**.

## 4. Measured NPU 0 Performance (`baseline` vs `tuned`)

Measured on host NPU 0 using `torch.npu.Event(enable_timing=True)` with `warmup = 5` and `repeats = 20`.

### 4.1 AttentionResidual

| Case | Shapes | Dtype | Baseline Median (ms) | Tuned Median (ms) | Baseline Mean (ms) | Tuned Mean (ms) | Speedup (Median) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `level4/attention_residual_1` | `[4, 1, 128, 2048]` | `bfloat16` | 0.0354 | 0.0246 | 0.0373 | 0.0252 | 1.44x |
| `level4/attention_residual_7` | `[4, 4, 512, 4096]` | `bfloat16` | 0.5585 | 0.3143 | 0.5594 | 0.3144 | 1.78x |
| `level4/attention_residual_16` | `[8, 4, 2048, 4096]` | `bfloat16` | 4.0389 | 2.1674 | 4.0398 | 2.1681 | 1.86x |
| `level4/attention_residual_18` | `[4, 8, 512, 7168]` | `bfloat16` | 1.6894 | 0.9418 | 1.6898 | 0.9415 | 1.79x |
| `level4/attention_residual_20` | `[32, 1, 1024, 2048]` | `bfloat16` | 1.2847 | 0.6742 | 1.2860 | 0.6743 | 1.91x |

### 4.2 MLA (CANNBench & Supplemental Cases)

| Case | Q_nope Shape | K_nope Shape | Layout / Causal / Dtype | Baseline Median (ms) | Tuned Median (ms) | Baseline Mean (ms) | Tuned Mean (ms) | Speedup (Median) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `level4/mla_1` | `[2, 128, 128, 512]` | `[2, 128, 1, 512]` | `BSND` / causal / `float16` | 1.1520 | 1.1652 | 1.1530 | 1.1683 | 0.99x |
| `level4/mla_2` | `[4, 256, 128, 512]` | `[4, 256, 1, 512]` | `BSND` / causal / `bfloat16` | 8.7589 | 8.6504 | 8.7590 | 8.6503 | 1.01x |
| `level4/mla_8` | `[16, 1, 128, 512]` | `[16, 2048, 1, 512]` | `BSND` / non-causal / `float16` | 0.9883 | 0.9865 | 0.9898 | 0.9863 | 1.00x |
| `level4/mla_11` | `[128, 1, 128, 512]` | `[128, 1024, 1, 512]` | `BSND` / non-causal / `float16` | 3.3388 | 3.3410 | 3.3390 | 3.3411 | 1.00x |
| `level4/mla_13` | `[2, 128, 128, 512]` | `[2, 1, 128, 512]` | `BNSD` / causal / `float16` | 1.1687 | 0.9697 | 1.1692 | 0.9702 | 1.21x |
| `level4/mla_15` | `[2, 32, 64, 512]` | `[2, 1, 512, 512]` | `BNSD` / causal / `bfloat16` | 0.4839 | 0.4846 | 0.4845 | 0.4845 | 1.00x |
| `supplemental/mla_gqa_bsnd_kv_tail` | `[1, 3, 64, 448]` | `[1, 129, 2, 448]` | `BSND` / causal / `bfloat16` | 0.0565 | 0.0565 | 0.0566 | 0.0562 | 1.00x |
| `supplemental/mla_gqa_decode_shard448` | `[1, 1, 128, 448]` | `[1, 128, 4, 448]` | `BSND` / non-causal / `float16` | 0.0402 | 0.0401 | 0.0403 | 0.0397 | 1.00x |
