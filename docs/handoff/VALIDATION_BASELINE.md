# VALIDATION_BASELINE — 独立复跑当前测试面与环境事实（不碰设备）

- 任务: shared task task-5 (VALIDATION-BASELINE)，执行者 teammate `flex-3-g1`
- 执行时间: 2026-11-12 晚（本地 Asia/Shanghai），只读调查阶段
- 代码基准（本报告全部分析所依据）: `git rev-parse HEAD` = `f341e47529453bdbc8dd81f2c108d36628fb4b8b`
  - tag `handoff-20261112` = `374de77`（docs: add next-session plan）；HEAD 在 tag 之上 +2 提交：
    `314e016`（bind AR/MLA outputs as trailing ABI parameters）、`f341e47`（identity over actual Op attributes and real Var binders）
  - 下文所有数字除特别标注外均为 **post-f341e47（当前 HEAD）实测**；pre-f341e47 的 legacy 树数字**未复跑（UNKNOWN）**。mainline 面在 f341e47 之前为 12 passed 系 Lead 实测转述，本执行者未独立验证。
- 边界: 本报告全程**未触碰设备**；npu-smi 仅被动只读查询；无 NPU kernel 执行、无编译、无设备申请；无 git 写操作。

---

## 1. 结论速览

| 项 | 结论 |
| --- | --- |
| 活动测试面（targets/ascend/tests_native_mainline） | **24 collected / 24 passed / 0 failed，退出码 0**（0.28s） |
| legacy tests/ 全量 | 231 failed / 759 passed / 5 skipped / 67 errors，退出码 1（18.89s） |
| tests/ 能否作为本任务回归基线 | **不能**。预存收集失败（环境 PYTHONPATH/依赖缺口）+ API 漂移（旧表面测试）占主导；tests/core 725 通过可作弱参考，tests/nvidia·apple·tilefoundry 失败与环境缺口及旧 API 强耦合 |
| 环境 | 常规依赖齐备（z3/tilelang/torch/ortools）；torch_npu 需 source CANN 后才可导入（裸导入缺 libhccl.so）；safetensors/transformers **未安装**；CANN 9.2.0 脚本与比较器脚本均在 |
| 设备 | 3×Ascend950PR，Health 全 OK（npu-smi 只读）；设备锁 root:root 不可写（普通用户） |
| 本次是否触碰设备 | **否** |
| 数值 40 例 | **NOT_RUN** |
| PASS 外推 | 本报告任何 PASS 均不得外推为设备或阶段 PASS |

---

## 2. 复跑交接验证命令（mainline）

命令（原样执行）:

```text
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
/home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests_native_mainline
```

- 结果: **24 passed in 0.28s，退出码 0**（`--collect-only -q` 同样给出 `24 tests collected`）
- 构成（按文件）:
  - `targets/ascend/tests_native_mainline/backend_operators/test_operator_abi_numeric.py` — 7 例（ABI 尾随输出绑定、值只落声明的输出、篡改拒绝、形状/dtype/非连续拒绝）
  - `targets/ascend/tests_native_mainline/test_identity.py` — 4 例
  - `targets/ascend/tests_native_mainline/test_native_builders.py` — 9 例
  - `targets/ascend/tests_native_mainline/test_search_rank.py` — 4 例
- 与 f341e47 之前的关系: Lead 通知活动面 12 → 24 passed（新增 test_identity.py + backend_operators 两个文件）。本执行者实测 24/24 通过，**与该描述一致**；12 处 raise MainlineNotImplemented 属 Lead 转述，未由本执行者复数（UNKNOWN）。
- 该套件为纯宿主侧（identity/ABI/构建器契约/search 排序），**不执行设备代码**。

---

## 3. legacy tests/ 刻画

命令:

```text
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
/home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider \
  --continue-on-collection-errors --tb=line -rA tests/
```

汇总（post-f341e47）: **231 failed, 759 passed, 5 skipped, 67 errors in 18.89s；EXIT_CODE=1**

### 3.1 分组计数

| 分组 | passed | failed | skipped | errors（整文件收集失败） |
| --- | --- | --- | --- | --- |
| tests/core | 725 | 2 | 0 | 1 |
| tests/nvidia | 21 | 201 | 5 | 62 |
| tests/tilefoundry | 13 | 25 | 0 | 1 |
| tests/apple | 0 | 3 | 0 | 3 |
| 合计 | 759 | 231 | 5 | 67 |

skipped 明细（全在 nvidia，模块级 skip 理由）:
- `SKIPPED [1] tests/nvidia/cuda/runtime/resource/test_resource_device.py:39` — "tracking a later set_device needs a second card to move to"
- `SKIPPED [4] tests/nvidia/cuda/scaffold/test_scaffold.py:18` — "requires CUDA"

### 3.2 67 个收集错误 = 整文件 import 失败（按文件去重: apple 3 / core 1 / nvidia 62 / tilefoundry 1）

| 根因 | 文件数 | 证据 |
| --- | --- | --- |
| `No module named 'tilefoundry_nvidia'` | 59 | 样例 `tests/nvidia/cuda/analysis/test_analysis_closure.py:19`（`from tilefoundry_nvidia import CudaTarget`）；包实际在 `targets/nvidia/tilefoundry_nvidia/`，而交接命令的 `PYTHONPATH=src:targets/ascend` 未含 `targets/nvidia` |
| `No module named 'tilefoundry_apple'` | 3 | `tests/apple/amx/target/test_amx_target.py:15`；包实际在 `targets/apple/tilefoundry_apple/`，PYTHONPATH 同样缺 `targets/apple` |
| `No module named 'safetensors'` | 3 | `tests/core/runtime/resource/test_resource_device.py`、`tests/nvidia/cuda/cli/test_cli_check.py`、`tests/nvidia/cuda/integration/models/deepseek_v4_flash/test_causal_lm_e2e.py`（依赖未安装，见 §4） |
| `No module named 'transformers'` | 1 | `tests/nvidia/cuda/integration/models/qwen3_5_35b_a3b/test_provenance.py` |
| `ImportError: cannot import name 'PIPELINE_INTENT_SCHEMA'` | 1 | `tests/tilefoundry/test_pipeline_intent_dce.py:28-31` 导入该符号；当前实现 `src/tilefoundry/parser/pipeline_intents.py:14-19` 的 `__all__` 只含 PipelineIntent / PipelineIntentSet / register_pipeline_intents / verify_pipeline_intents，**符号已不存在**（模块文件本身存在且被 git 跟踪） |

### 3.3 231 个 failed 的根因抽样（--tb=line 汇总与代表性证据）

tb=line 根因类型分布（全部失败行）: AssertionError ×128、ValueError ×76、TypeError ×12、`tilefoundry.ir.hir.pipeline.PipelineError` ×11、其他少量。

- **tests/nvidia（201 failed）**: 大量失败经由共享 corpus 帮助器 `tests/core/_corpus.py:101` 断言，报错形态 "…imports 'tilefoundry_nvidia', which is not importable."——即与收集错误同源的 **PYTHONPATH 缺 targets/nvidia**；其余为 API 漂移（例: vendor 缺失拒绝信息、scaffold/weight-loading/mesh 期望旧行为）。涉及已收集模块: parser/mesh_scopes、runtime/resource/child_module_resources、schedule/storage_context、scaffold 等。
- **tests/tilefoundry（25 failed）**: pipeline 表面 API 变更——`tests/tilefoundry/test_pipeline_surface.py:74` `TypeError: PipelineSpec.__init__() got an unexpected keyword argument 'num_stages'`；`tests/tilefoundry/test_dce_effect_aware.py:120` 报 `PIPELINE_ANNOTATION_UNSUPPORTED: unsupported tf.pipelined keyword 'num_stages'; accepts ('buffer_schedules', 'enable_offset', 'fixed_ii', 'max_buffer_versions', 'max_stage', 'task_intents')`——legacy 测试仍按旧 num_stages 表面编写。
- **tests/core（2 failed）**:
  - `tests/core/cli/test_cli.py:85` — 期望描述文本 "TileFoundry — A tile-based, agentic platform"，实际为 "TileFoundry — TileFoundry"（元数据文本漂移）。
  - `tests/core/scripts/test_machine_paths_lint.py:123` — 机器路径 lint：`ACTIVE_CONTRACT.md:5` 引用 `/home/s00653124` 下工作树路径（`TileFoundry_Ascend950PR_AgentTeam/...`），在本机不存在 → **环境性失败**（交接文档携带的历史机器路径），非代码回归。
- **tests/apple（3 failed）**: 运行期目标注册缺失——`tests/apple/amx/cli/test_cli.py:12` 报 `unknown target identity 'apple.m2_pro'; available: ['cpu', 'vendor.npu']`；`tests/apple/amx/target/test_target.py:9` 断言 `{'amx'} <= registry` 失败。根因同收集错误: `tilefoundry_apple` 不可导入 → amx/apple 目标从未注册。

### 3.4 结论: tests/ 能否作为本任务的回归基线

**不能作为本任务（Ascend950PR 交接）的回归基线**，理由:
1. 67/约 1059 项结果为整文件收集失败，其根因是**环境口径差异**（PYTHONPATH 只含 `src:targets/ascend`，缺 `targets/nvidia`、`targets/apple`；safetensors/transformers 未安装）而非被测代码缺陷。补齐 PYTHONPATH 后这些文件可能大量转绿，但这属于**另一环境口径**，不属于交接验证命令的口径。
2. 显著比例失败（nvidia 201、tilefoundry 25、apple 3）与 **API 漂移**强耦合（pipeline 表面 num_stages → max_stage/buffer_schedules 家族、PIPELINE_INTENT_SCHEMA 符号移除、CLI 描述文本、目标注册面），即 legacy 套件滞后于当前快照 API。
3. 即使是 725 通过的 tests/core，也混有环境性失败（machine_paths_lint）。因此 legacy 树的 pass/fail 数字**不可复现地**依赖环境口径，作为基线会把"环境缺口/旧 API"与"新回归"混在一起。
4. 建议: 本任务的回归基线采用 §2 的 `targets/ascend/tests_native_mainline`（24 passed，host 侧、无设备依赖）；tests/core 的 725 通过可作为弱参考信号，其余分组仅做根因跟踪，不纳入回归门。

---

## 4. 环境事实（只读）

| 项 | 值 | 备注/证据 |
| --- | --- | --- |
| Python | 3.12.16，`/home/s00653124/workspace/envs/tfnpu-v2/bin/python` | venv `sys.base_prefix` = `/home/s00653124/.conda/envs/tilefoundry-npu-dev`（stdlib 共享该 conda 环境 python3.12） |
| z3 | 4.15.4，可导入 | site-packages/z3 |
| tilelang | `0.1.15+cpu.git0e687c17`，可导入 | 来源: 本地 wheel `file:///home/s00653124/workspace/envs/wheels-main/tilelang-0.1.15%2Bcpu.git0e687c17-cp39-abi3-linux_x86_64.whl`（direct_url.json，sha256 d34ed401f2b51f7613cf81185a32688e74488f43c6cf3223c2ee7ddd95edb9ac）；git 0e687c17；**CPU 构建** |
| torch | 2.12.0+cpu，可导入 | site-packages/torch |
| torch_npu | 2.12.0 | **裸导入失败**: `ImportError: libhccl.so: cannot open shared object file`；source `/data/pri/cann-9.2.0/ascend-toolkit/set_env.sh` 后可导入成功。导入时警告 `/data/pri/cann-9.2.0/cann-9.2.0-beta.2` 属主与当前用户不匹配 |
| ortools | 9.15.6755，可导入 | site-packages/ortools |
| safetensors / transformers | **均未安装**（find_spec None） | 对应 §3.2 的 4 个收集错误 |
| CANN 环境脚本 | `/data/pri/cann-9.2.0/ascend-toolkit/set_env.sh` 存在 | symlink → `latest/set_env.sh` |
| 比较器 | `/home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts/npu_compare.py` 存在 | 9041 字节，属主 s00653124，mtime Nov 5 20:21 |
| 设备锁 | `/home/.tf_device0.lock` 存在 | 权限 `rw-rw-r-- root:root`，0 字节，mtime Nov 12 09:59；普通用户无写权限（仅 root/root 组可写） |
| NPU | npu-smi 25.7.rc1 只读查询: **3×Ascend950PR，Health 全 OK** | 利用率 0%；HBM 7200 / 5425 / 5639 of 131072 MB；三卡均有**他人 python 进程驻留**（PID 3103591、3091897 于 NPU0；3148145 于 NPU1；3148233 于 NPU2）——本执行者未申请、未占用任何卡 |

---

## 5. 边界声明（task-5 要求逐项）

1. **本次是否触碰设备 = 否**。全程未申请设备、未编译、未 launch kernel；npu-smi 为被动状态查询（只读）。
2. **数值 40 例 = NOT_RUN**。未执行任何设备侧数值比对（mainline 24 例与 legacy 通过例均为 host 侧逻辑/契约测试，不构成数值验证）。
3. **任何 PASS 不得外推为设备或阶段 PASS**。§2 的 24 passed 仅证明 host 侧契约；§3 的 759 passed 属 legacy host 侧行为。
4. pre-f341e47 的 legacy 计数未复跑（UNKNOWN）；本报告数字为 post-f341e47 实测。
5. 遗留设备锁与驻留进程说明: NPU0/1/2 上已有其他会话的 python 进程（见 §4），申请设备前须先核查锁与占用（由设备申请流程处理，非本任务范围）。

---

## 6. 运行合规记录

- pytest 均带 `-p no:cacheprovider`；未创建 worktree；无 git 写操作（仅 `git rev-parse/status/log/describe` 只读）。
- 落盘: 仅本报告 `docs/handoff/VALIDATION_BASELINE.md` 与 /tmp 下两份原始日志（`/tmp/task5_mainline.log`、`/tmp/task5_legacy.log`；/tmp 不保证持久）。未改 src/、targets/、tests/。
- 运行期写入仓库的只有 conftest 固有的 gitignored 产物根（`test_results/`、`.torch_extensions/`，见 `.gitignore:231-232`；`.pytest_cache/` 因 no:cacheprovider 未生成）。`git status` 复核: 除本报告目录 `docs/handoff/` 与并行团队的 `docs/mainline_validation/*` untracked 产物外，无其他改动。
- 分析时间窗内并行团队提交了 314e016、f341e47；本报告全部数字在 f341e47 工作树上取得。
