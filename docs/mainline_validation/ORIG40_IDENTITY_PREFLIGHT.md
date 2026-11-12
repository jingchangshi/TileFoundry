# task-112 / ORIG-40：原 fixture/runner/comparator 身份与 preflight

> 只读产出，explorer-g2，UTC 2026-11-12T12:35:07.747Z。范围：原 40 例身份、包完整性、CPU 测试、比较器身份和两算子 connected preflight-only。未导入候选、未调用 test.main、未启动 kernel、未同步设备、未运行数值 40。

## 1. 总状态

| 项 | 状态 |
|---|---|
| package integrity | PASS（before 与 after 均退出 0 且结果一致） |
| framework CPU tests | 35 passed |
| reference CPU tests | 20 passed |
| run_contract CLI | --help 退出 0 |
| AR preflight-only | PASS，numerical=NOT_RUN |
| MLA preflight-only | PASS，numerical=NOT_RUN |
| original 40 numerical | **NOT_RUN** |
| device kernel | **NOT_RUN** |
| candidate import | false |
| test.main invocation | false |
| device synchronize | false |

## 2. 执行命令与退出码

| name | exit | cwd | command |
|---|---:|---|---|
| verify_package_before | 0 | `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3` | `PYTHONDONTWRITEBYTECODE=1 /home/s00653124/workspace/envs/tfnpu-v2/bin/python tools/verify_package.py --output /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/native_v3/mainline_validation/verify_package.json` |
| framework_tests_35 | 0 | `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3` | `PYTHONDONTWRITEBYTECODE=1 /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m unittest discover -s tests -v` |
| reference_tests_20 | 0 | `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3` | `PYTHONDONTWRITEBYTECODE=1 /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m unittest discover -s reference_core -p 'test_*.py' -v` |
| run_contract_help | 0 | `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3` | `PYTHONDONTWRITEBYTECODE=1 /home/s00653124/workspace/envs/tfnpu-v2/bin/python tools/run_contract.py --help` |
| preflight_attention_residual | 0 | `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3` | `source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh && PYTHONPATH=/home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 flock -w 600 /home/.tf_device0.lock /home/s00653124/workspace/envs/tfnpu-v2/bin/python tools/run_contract.py --case attention_residual --operator-file /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/implementation/__init__.py --preflight-only --cache-dir off --report /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/native_v3/mainline_validation/preflight_attention_residual.json --log /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/native_v3/mainline_validation/preflight_attention_residual.log` |
| preflight_mla | 0 | `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3` | `source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh && PYTHONPATH=/home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 flock -w 600 /home/.tf_device0.lock /home/s00653124/workspace/envs/tfnpu-v2/bin/python tools/run_contract.py --case mla --operator-file /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/implementation/__init__.py --preflight-only --cache-dir off --report /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/native_v3/mainline_validation/preflight_mla.json --log /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/native_v3/mainline_validation/preflight_mla.log` |
| verify_package_after | 0 | `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3` | `PYTHONDONTWRITEBYTECODE=1 /home/s00653124/workspace/envs/tfnpu-v2/bin/python tools/verify_package.py --output /home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/native_v3/mainline_validation/verify_package_after.json` |

所有 Python 命令均带 `PYTHONDONTWRITEBYTECODE=1`；两个 preflight 另带 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、CANN92 `source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh`、比较器 `PYTHONPATH`，并使用 `flock -w 600 /home/.tf_device0.lock`。

## 3. 包与 runner 身份

- 包根：`/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3`
- verify_package：fixture_files=18，original_cases=40，tasks=34，gates=8，python_files_syntax_checked=30，package_manifest_checked=true
- PACKAGE_MANIFEST：82 份文件

| 文件 | sha256 |
|---|---|
| `tools/run_contract.py` | `cff4e08b0f4f0b3bb3c5325a473af5d554b684a12b04afe611bd1b3d1d9d807e` |
| `tools/verify_package.py` | `0d554dc153980e1feed4de0a2faaf87919c7b23c21d40fff5002796da5f16484` |
| `cases/case_matrix.json` | `d57785b1db4205e2f11a0c1641eed7cddbd0b6530d81655a558dcdf556be4ef8` |
| `cases/CONTRACT.md` | `3fb9f41454e36ae9f462a7c3e69ec3028aa4b602fed0687951e9df5165ceda92` |
| `fixtures/source_manifest.json` | `2548c4670c4883584965df4b5c9dcad819e9c16873fd689ac43379b69506636d` |
| `PACKAGE_MANIFEST.json` | `d029589a67558bf0a989f5a151f6dddd29b5a8a9f1ce71cacb62bca16e56a530` |
| `implementation/__init__.py` | `159f7088440c23acdd09f9741e8f89fb581f66bd35d997bd2f40acf2c1a930b2` |

### 18 份原 fixture 身份

| path | bytes | sha256 |
|---|---:|---|
| `fixtures/attention_residual/golden.py` | 2473 | `cf90b8476c4fabbc5be348baf48b44c0ce63ff44798f5479100232762217795a` |
| `fixtures/attention_residual/desc.md` | 6455 | `5eb025defbff0498d7f41f23e967080eae022dbb8adffd969aaf35c6a61c1c25` |
| `fixtures/attention_residual/proto.yaml` | 2631 | `2d85e237072501be7205cf85c82739d4b1be17a37c74fe80acf1f857db19aa73` |
| `fixtures/attention_residual/cases.yaml` | 8284 | `d6d35f240ab979fce0f7e8ae35c27ab1fd6bb00213d0bcbd65e9ce28dcc16792` |
| `fixtures/attention_residual/cases.csv` | 4868 | `eddae5a97b9d08986f76899bbfe88dfdac501b5481ebff50bfd40fbaae5d5b8a` |
| `fixtures/attention_residual/test_attention_residual.py` | 571 | `c50fcf953ffe613f743dc4c34b5f715676cca6f491b0d37bff8946959d429559` |
| `fixtures/attention_residual/bench_attention_residual.py` | 613 | `5a02e2c40d41a364eac1b5b6cac811cdebf65dd1412e59a08652b9bfb891e6c8` |
| `fixtures/attention_residual/LICENSE` | 5278 | `cb282ac2b66d7d6734c3fb354db792b28cf23ab29fbc16d5cfd8bd542bedeb1e` |
| `fixtures/mla/golden.py` | 5939 | `3996ec1a4bcf142db4f311dd72510d0420269691e1efccbd72a00e53a2bc8cdc` |
| `fixtures/mla/desc.md` | 13322 | `433b8eaff3231700cb2fe0ecbba8cc91cb2ae65902517d1a4aa66daf48d10362` |
| `fixtures/mla/proto.yaml` | 3520 | `70cd11633ebad052058cc33606f4fa4e004fc1e9b2a9251a73f28a697a1910f4` |
| `fixtures/mla/cases.yaml` | 10983 | `57e742e5f14680996c348cdfcb462852e64750066b4d4f4c852c4403dfbdf46d` |
| `fixtures/mla/cases.csv` | 7542 | `85648d43619c1a38c5730231baa20308dc5166c76d3cbbb8fe0f53ab7d2045a8` |
| `fixtures/mla/test_mla.py` | 526 | `f12622752c0d3f6d2232b4fa10aef332b1b1dc8e7f708e15b4794d480408d981` |
| `fixtures/mla/bench_mla.py` | 523 | `f2f25892640692960d6fcc0b587735319a262c161910a3f66551a9afe6efa431` |
| `fixtures/mla/LICENSE` | 5278 | `cb282ac2b66d7d6734c3fb354db792b28cf23ab29fbc16d5cfd8bd542bedeb1e` |
| `fixtures/cannbench_contract.py` | 52173 | `026f51a94b6a6046f3096fcaa363f77e325e7f246c6e584a65a44e80de457400` |
| `fixtures/CANN_CONTRACT_LICENSE` | 5278 | `cb282ac2b66d7d6734c3fb354db792b28cf23ab29fbc16d5cfd8bd542bedeb1e` |

原入口为 `attention_residual(v_stack,x,w,gamma,alpha,epsilon=1e-6)` 与 `mla(q_nope,q_rope,k_nope,k_rope,v,numKVHeads=1,scaleValue=-1.0,inputLayout="BSND",is_causal=False)`；原测试入口为 `main(operator_api)`。preflight 仅验证入口存在，不调用。

## 4. 比较器身份与阈值来源

- 现用比较器：`/home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts/npu_compare.py`
- 仓库：BiShengKernelAgent `5496ef6e342da64386e0dfb4f0d8b91eb3712b2f`，分支 `feat/tla-gemm`，状态 `clean`
- 原测试通过 `from npu_compare import compare_case_result`，以 `accuracy_mode="cann-bench"` 调用；`run_contract.py` 记录同一 entrypoint。
- 阈值来源：`/home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts/cann_bench_compare.py:82-132`（安装比较器内部表）；helper/proto 的 `precision_thresholds` 为空。本次未改阈值、未替换比较器、未自行推断 allclose。

| comparator file | sha256 |
|---|---|
| `npu_compare.py` | `ef2506bba60a3126196f433d97a02455c96e7be9c6c689b0a7ff400b97af5211` |
| `cann_bench_compare.py` | `e347b92141572e7c987e39296806a6483c63663e729917f50113c357e82cf3fc` |
| `dtype_close_compare.py` | `ab2cdf48fe5702c13bef3c021b2f62d5f1d5e855b9097cae5f34f5fa1f6cd37e` |
| `npu_compare_common.py` | `59a22dad8ea8769cbe742720c2fb5c4b1be539abccc4a72cf22468333a7b5305` |
| `npu_contract_compare.py` | `63e6d934c850feb9e9a8104c3ffd6398e5f865e871fabcea8922b12dab699dc3` |

### 安装比较器内建阈值表

| dtype | threshold | small_value_threshold | small_value_error | cancel_boundary | cancel_zero_threshold |
|---|---:|---:|---:|---:|---:|
| float16 | 2^-10 | 2^-11 | 2^-16 | 2^-5 | 2^-5 |
| bfloat16 | 2^-7 | 2^-8 | 2^-16 | 2^-3 | 2^-3 |
| float32 | 2^-13 | 2^-14 | 2^-30 | 2^-8 | 2^-8 |
| float64 | 2^-13 | 2^-14 | 2^-30 | 2^-8 | 2^-8 |
| hifloat32 | 2^-11 | 2^-12 | 2^-28 | 2^-8 | 2^-8 |
| float8_e4m3 | 2^-3 | 2^-4 | 2^-6 | 2^-1 | 2^-1 |
| float8_e5m2 | 2^-2 | 2^-3 | 2^-5 | 1.0 | 1.0 |

## 5. preflight-only 结果

| case | run_id | preflight | numerical | device | cache | expected | executed | records | passed | candidate_imported | main | kernel_requested | source_changed |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| attention_residual | `d44d0d786db64f77aaed3acb490297ce` | PASS | NOT_RUN | npu:0 / Ascend950PR_9579 | off | 20 | 0 | 0 | 0 | false | false | false | 0 |
| mla | `ba328d3967674ae4a6bd18e6bacbe97f` | PASS | NOT_RUN | npu:0 / Ascend950PR_9579 | off | 20 | 0 | 0 | 0 | false | false | false | 0 |

两个 preflight 均确认 `candidate_imported=false`、`contract_main_invoked=false`、`kernel_execution_requested=false`、`main_returned_normally=false`、`device_synchronize_succeeded=false`；`--operator-file` 只作为 CLI 必填文件身份占位，不是候选，也没有被导入。CANN/torch_npu 输出的 owner warnings 非阻断。

## 6. 原 40 label 与输入矩阵

| # | case | label | seed | shapes | dtypes | attrs | note |
|---:|---|---|---:|---|---|---|---|
| 1 | attention_residual | `level4/attention_residual_1` | 1474891499 | `[[4,1,128,2048],[1,128,2048],[2048],[2048]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [P:default-small] warm probe |
| 2 | attention_residual | `level4/attention_residual_2` | 745556439 | `[[2,1,1,4096],[1,1,4096],[4096],[4096]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [B:T1-decode-Nmin] single-token decode, minimum stack |
| 3 | attention_residual | `level4/attention_residual_3` | 1466990822 | `[[6,2,256,2048],[2,256,2048],[2048],[2048]]` | `float16,float16,float16,float16` | `{"alpha":0.2,"epsilon":0.000001}` | [D:fp16-S] fp16 small |
| 4 | attention_residual | `level4/attention_residual_4` | 174300161 | `[[3,1,333,2047],[1,333,2047],[2047],[2047]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [B:non-aligned] N3/T333/D2047 |
| 5 | attention_residual | `level4/attention_residual_5` | 1166651234 | `[[8,1,512,1024],[1,512,1024],[1024],[1024]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.95,"epsilon":0.000001}` | [B:alpha-max] aggregation-dominated blend |
| 6 | attention_residual | `level4/attention_residual_6` | 1917756235 | `[[8,2,1024,2048],[2,1024,2048],[2048],[2048]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.35,"epsilon":0.000001}` | [M:7B-block8] mid prefill |
| 7 | attention_residual | `level4/attention_residual_7` | 1546319123 | `[[4,4,512,4096],[4,512,4096],[4096],[4096]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [M:7B-batch4] |
| 8 | attention_residual | `level4/attention_residual_8` | 972998724 | `[[16,1,512,4096],[1,512,4096],[4096],[4096]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [M:deep-stack16] |
| 9 | attention_residual | `level4/attention_residual_9` | 1498121861 | `[[4,1,4096,2560],[1,4096,2560],[2560],[2560]]` | `float16,float16,float16,float16` | `{"alpha":0.5,"epsilon":0.000001}` | [D:fp16-M] long-seq fp16 |
| 10 | attention_residual | `level4/attention_residual_10` | 1196037518 | `[[12,2,512,3584],[2,512,3584],[3584],[3584]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.65,"epsilon":0.000001}` | [M:8B-block12] |
| 11 | attention_residual | `level4/attention_residual_11` | 99796731 | `[[6,1,2048,4096],[1,2048,4096],[4096],[4096]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [M:7B-seq2k] |
| 12 | attention_residual | `level4/attention_residual_12` | 1674213678 | `[[8,8,256,2048],[8,256,2048],[2048],[2048]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.05,"epsilon":0.000001}` | [B:alpha-min] residual-dominated blend |
| 13 | attention_residual | `level4/attention_residual_13` | 622278726 | `[[4,2,1024,5120],[2,1024,5120],[5120],[5120]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":1e-8}` | [B:eps-min] |
| 14 | attention_residual | `level4/attention_residual_14` | 1634482190 | `[[8,1,2048,1000],[1,2048,1000],[1000],[1000]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.001}` | [B:eps-max+non-aligned-D1000] |
| 15 | attention_residual | `level4/attention_residual_15` | 1676476532 | `[[24,1,256,4096],[1,256,4096],[4096],[4096]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [G:deep-stack24] |
| 16 | attention_residual | `level4/attention_residual_16` | 1722815757 | `[[8,4,2048,4096],[4,2048,4096],[4096],[4096]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [M:op-max] stack at 2^28 cap |
| 17 | attention_residual | `level4/attention_residual_17` | 987619029 | `[[16,1,4096,4096],[1,4096,4096],[4096],[4096]]` | `float16,float16,float16,float16` | `{"alpha":0.5,"epsilon":0.000001}` | [D:fp16-op-max] deep variant at cap |
| 18 | attention_residual | `level4/attention_residual_18` | 1443277225 | `[[4,8,512,7168],[8,512,7168],[7168],[7168]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.25,"epsilon":0.000001}` | [M:DSv3-width] D=7168 |
| 19 | attention_residual | `level4/attention_residual_19` | 605202302 | `[[8,1,8192,2560],[1,8192,2560],[2560],[2560]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [M:long-seq-8k] |
| 20 | attention_residual | `level4/attention_residual_20` | 711900270 | `[[32,1,1024,2048],[1,1024,2048],[2048],[2048]]` | `bfloat16,bfloat16,bfloat16,bfloat16` | `{"alpha":0.5,"epsilon":0.000001}` | [G:deep-stack32] |
| 21 | mla | `level4/mla_1` | 1046154938 | `[[2,128,128,512],[2,128,128,64],[2,128,1,512],[2,128,1,64],[2,128,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":true}` | M-float16-BSND-B2S128Nq128Nkv1-dnope512drope64-prefill-causal \| baseline_kernels: FusedInferAttentionScore×1 |
| 22 | mla | `level4/mla_2` | 1466203925 | `[[4,256,128,512],[4,256,128,64],[4,256,1,512],[4,256,1,64],[4,256,1,512]]` | `bfloat16,bfloat16,bfloat16,bfloat16,bfloat16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":true}` | L-bfloat16-BSND-B4S256Nq128Nkv1-dnope512drope64-prefill-causal \| baseline_kernels: FusedInferAttentionScore×1 |
| 23 | mla | `level4/mla_3` | 1481725773 | `[[2,128,64,448],[2,128,64,64],[2,128,1,448],[2,128,1,64],[2,128,1,448]]` | `bfloat16,bfloat16,bfloat16,bfloat16,bfloat16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":true}` | M-bfloat16-BSND-B2S128Nq64Nkv1-dnope448drope64-causal |
| 24 | mla | `level4/mla_4` | 625327224 | `[[2,256,128,512],[2,256,128,64],[2,128,1,512],[2,128,1,64],[2,128,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | M-float16-BSND-B2Sq256Skv128Nq128-dnope512-cross \| baseline_kernels: FusedInferAttentionScore×1 |
| 25 | mla | `level4/mla_5` | 1524641592 | `[[2,256,128,512],[2,256,128,64],[2,256,1,512],[2,256,1,64],[2,256,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":true}` | L-float16-BSND-B2S256Nq128-dnope512-long-prefill-causal \| baseline_kernels: FusedInferAttentionScore×1 |
| 26 | mla | `level4/mla_6` | 1771309450 | `[[2,256,128,448],[2,256,128,64],[2,256,1,448],[2,256,1,64],[2,256,1,448]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | L-float16-BSND-B2S256Nq128-dnope448-long-prefill |
| 27 | mla | `level4/mla_7` | 1475820401 | `[[1,1,128,512],[1,1,128,64],[1,2048,1,512],[1,2048,1,64],[1,2048,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | S-float16-BSND-B1S1Skv2048Nq128-decode-single-batch \| baseline_kernels: FusedInferAttentionScore×1 |
| 28 | mla | `level4/mla_8` | 1124036600 | `[[16,1,128,512],[16,1,128,64],[16,2048,1,512],[16,2048,1,64],[16,2048,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | M-float16-BSND-B16S1Skv2048Nq128-decode \| baseline_kernels: FusedInferAttentionScore×1 |
| 29 | mla | `level4/mla_9` | 253874537 | `[[32,1,128,512],[32,1,128,64],[32,2048,1,512],[32,2048,1,64],[32,2048,1,512]]` | `bfloat16,bfloat16,bfloat16,bfloat16,bfloat16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | L-bfloat16-BSND-B32S1Skv2048Nq128-decode \| baseline_kernels: FusedInferAttentionScore×1 |
| 30 | mla | `level4/mla_10` | 1151581541 | `[[60,1,128,512],[60,1,128,64],[60,2048,1,512],[60,2048,1,64],[60,2048,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | L-float16-BSND-B60S1Skv2048Nq128-dnope512-decode-large-batch \| baseline_kernels: FusedInferAttentionScore×1 |
| 31 | mla | `level4/mla_11` | 1138739301 | `[[128,1,128,512],[128,1,128,64],[128,1024,1,512],[128,1024,1,64],[128,1024,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | L-float16-BSND-B128S1Skv4096Nq128-dnope512-decode-large-batch \| baseline_kernels: FusedInferAttentionScore×1 |
| 32 | mla | `level4/mla_12` | 785373728 | `[[60,1,64,512],[60,1,64,64],[60,2048,1,512],[60,2048,1,64],[60,2048,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":false}` | L-float16-BSND-B60S1Skv2048Nq64-dnope512-decode-large-batch \| baseline_kernels: FusedInferAttentionScore×1 |
| 33 | mla | `level4/mla_13` | 1101264333 | `[[2,128,128,512],[2,128,128,64],[2,1,128,512],[2,1,128,64],[2,1,128,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BNSD","is_causal":true}` | M-float16-BNSD-B2Nq128S128-dnope512drope64-prefill-causal \| baseline_kernels: FusedInferAttentionScore×1 |
| 34 | mla | `level4/mla_14` | 1554677888 | `[[2,128,128,448],[2,128,128,64],[2,1,128,448],[2,1,128,64],[2,1,128,448]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BNSD","is_causal":true}` | M-float16-BNSD-B2Nq128S128-dnope448drope64-prefill-causal |
| 35 | mla | `level4/mla_15` | 471422046 | `[[2,32,64,512],[2,32,64,64],[2,1,512,512],[2,1,512,64],[2,1,512,512]]` | `bfloat16,bfloat16,bfloat16,bfloat16,bfloat16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BNSD","is_causal":true}` | [R:bounded-resource] dtype=bfloat16/bfloat16/bfloat16/bfloat16/bfloat16; shapes=2x32x64x512/2x32x64x64/2x1x512x512/2x1x512x64/2x1x512x512; inputLayout=BNSD,is_causal=True |
| 36 | mla | `level4/mla_16` | 571191437 | `[[60,128,1,512],[60,128,1,64],[60,1,2048,512],[60,1,2048,64],[60,1,2048,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BNSD","is_causal":false}` | L-float16-BNSD-B60Nq128S1Skv2048-decode-large-batch \| baseline_kernels: FusedInferAttentionScore×1 |
| 37 | mla | `level4/mla_17` | 1407169922 | `[[16,2,128,512],[16,2,128,64],[16,2048,1,512],[16,2048,1,64],[16,2048,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":true}` | L-float16-BSND-B16Sq2Skv2048Nq128-dnope512-MTP-causal \| baseline_kernels: FusedInferAttentionScore×1 |
| 38 | mla | `level4/mla_18` | 1153905209 | `[[4,2,32,512],[4,2,32,64],[4,512,1,512],[4,512,1,64],[4,512,1,512]]` | `bfloat16,bfloat16,bfloat16,bfloat16,bfloat16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":true}` | [R:bounded-resource] dtype=bfloat16/bfloat16/bfloat16/bfloat16/bfloat16; shapes=4x2x32x512/4x2x32x64/4x512x1x512/4x512x1x64/4x512x1x512; inputLayout=BSND,is_causal=True |
| 39 | mla | `level4/mla_19` | 203201392 | `[[16,128,2,512],[16,128,2,64],[16,1,2048,512],[16,1,2048,64],[16,1,2048,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BNSD","is_causal":false}` | L-float16-BNSD-B16Nq128Sq2Skv2048-dnope512-MTP \| baseline_kernels: FusedInferAttentionScore×1 |
| 40 | mla | `level4/mla_20` | 1898406220 | `[[1,256,128,512],[1,256,128,64],[1,256,1,512],[1,256,1,64],[1,256,1,512]]` | `float16,float16,float16,float16,float16` | `{"numKVHeads":1,"scaleValue":-1,"inputLayout":"BSND","is_causal":true}` | M-float16-B1-S256-去重替换 [R:replaced vr-only dup of c1] |

case_matrix 与两个 preflight 的 expected label 集合一致：AR=PASS，MLA=PASS。

## 7. NOT_RUN 边界

- CPU 35/20 测试不是原 40 数值合同，也不是 native/TL/NPU 验收。
- preflight-only 成功只证明依赖、fixture、入口、比较器和设备选择可用，不证明任何数值正确性。
- 本次没有运行任何原 40 case、没有 golden/oracle 计算、没有候选导入、没有 test.main、没有 kernel、没有同步、没有计时、没有性能测量。
- 没有 schema/plan/PassConfig/主链 conformance 结论；这些由 MAIN-C/D 的独立矩阵另行验证。
- 未改阈值、未替换比较器、未修改 fixture/runner/包源码；preflight 后包 verify_package 与 before 结果一致。

## 8. 结构化证据

- 本报告：`artifacts/native_v3/mainline_validation/ORIG40_IDENTITY_PREFLIGHT.md`
- JSON 汇总：`artifacts/native_v3/mainline_validation/orig40_identity_summary.json`
- verify before/after：`artifacts/native_v3/mainline_validation/verify_package.json`、`artifacts/native_v3/mainline_validation/verify_package_after.json`
- preflight reports/logs：`artifacts/native_v3/mainline_validation/preflight_attention_residual.json`、`.log`、`preflight_mla.json`、`.log`
- CPU logs：`artifacts/native_v3/mainline_validation/framework_tests.log`、`reference_tests.log`、`run_contract_help.log`

