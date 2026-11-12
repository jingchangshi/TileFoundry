# CHALLENGE-PROBES：fail-closed 与假通过调查

分析版本：
- 当前 HEAD（实测）：`f341e47529453bdbc8dd81f2c108d36628fb4b8b`，post-f341e47。
- 交接标签 `handoff-20261112`：`374de776f68d3e26eb80d9a1b7c8f67699708213`，pre-f341e47。
- 以上由只读 `git rev-parse HEAD handoff-20261112` 得到，exit 0。本报告实际运行的是当前 HEAD；未 checkout、未运行旧 worktree，也未将当前结果冒充为标签实测。
- 只写本报告及 [challenge_probes.py:1](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/probes/challenge_probes.py#L1>) 与 [challenge_results.jsonl:1](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/probes/challenge_results.jsonl#L1>)。没有设备调用、没有 git 写操作。Python 均使用指定解释器及三个环境变量；pytest 禁用 cacheprovider。

## 结论与失败/意外

1. **最终导出保持 fail-closed；未发现 ready=True。** [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) 无条件追加 NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED，[p3_chain.py:202](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L202>) 才计算 ready。所有完成的正常属性读取路径都被此处兜底。本结果不等于真实工件校验已经完成。
2. **高优先级：局部“真实工件”检查可整体伪造。** 匹配 identity 的 SimpleNamespace plan/completion/storage、NativeArtifact 包裹的任意非 TL 文本、假 parsed、假 ABI、假 conformance，配合自造 APPLIED transaction，缺口只剩上述无条件一项。属性齐全但非 ChainRecord 的 SimpleNamespace 也得到同样结果。若仅删 [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>)，从当前直线控制流推导这些输入会 ready=True；本调查没有修改实现来执行删行试验。
3. **高优先级：协调伪造 identity 可以只剩同一兜底。** dataclasses.replace 同时换掉 record 的 plan/completion/storage/tl identity，并同步假对象字段，且 artifact 的 source/tir/target/toolchain identity 随意填非空字符串，局部检查全过。单独把 record.plan_identity 改成普通对象则触发三处 mismatch，说明匹配性检查存在，但来源真实性未建立。
4. **健壮性缺陷：CHAIN_KEY 为 dict 抛 AttributeError**，原串为 `'dict' object has no attribute 'entry_identity'`，没有稳定 code。原因见 [p3_chain.py:173](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L173>)、[p3_chain.py:184](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L184>)：未先校验 record 类型。建议 CHAIN_RECORD_NOT_ACTUAL。
5. **主线 pass 的“函数返回成功”不是编译 PASS。** 最小函数和空模块均返回原 Module，IR identity 不变，但新增两个 metadata 项，ready=False，reason=MAINLINE_CHAIN_NOT_IMPLEMENTED；见 [p3_chain.py:221](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L221>)–243。六个直接阶段调用均 MainlineNotImplemented，输入 IR/metadata/参数 repr 在探针比较范围内无变化。
6. **正常入口强制消费 gate 尚不能宣称闭环。** 搜索 src 无 require_export_ready/EXPORT_GATE_KEY/export_gate 引用；target 唯一调用在 [p3_chain.py:242](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L242>)，且 raising=False。[finalize.py:18](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/schedule/finalize.py#L18>) 返回 AFTER_LOWER 结果，[compile.py:110](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/compile.py#L110>) 进入通用 build/link 路由。是否有其他后续拒绝保证 Ascend 永不发布 runnable 是 UNKNOWN；本任务禁止设备，未调用 build/compile 去触发临时目录或 runtime。此点与“直接 gate 永远 False”应区分。

## 鸭子类型接缝与唯一拦截位置

下表输入都在真实 Module + PrimFunction 上运行。这里“唯一兜底”指当其他字段按脚本 candidate() 协调后，完整 gaps 只有 NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED。并不表示空字段、dict、错误身份也能过。

| 接缝 | 可伪造输入 | 局部是否拦截 | 最终唯一拦截 / 较硬检查 |
|---|---|---|---|
| ChainRecord | NS(**record.as_dict())，非 dataclass | 不拦截；没有 isinstance(ChainRecord) | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| plan | NS(identity='p') | 不拦截；[p3_chain.py:80](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L80>) 取属性后 str；[p3_chain.py:115](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L115>) 只拒绝 dict/空 identity | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| completion | NS(identity='c')；无需 CompletionPlan/verified | 不拦截；同 [p3_chain.py:105](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L105>)–118 | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| storage | NS(identity='s')；无需 StoragePlan/verified/capacity | 不拦截；同 [p3_chain.py:105](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L105>)–118 | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| NativeArtifact 外壳 | 真实 NativeArtifact 内塞假内容 | 外壳有类型检查；内容不保证真实生成 | dict 由 [p3_chain.py:126](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L126>) 拦，合法外壳仅 [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| source | 'arbitrary non-TL text' | 只查 str + strip 非空（[p3_chain.py:128](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L128>)），未 parse/重算 source digest | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| ABI | NS(input_count=0,output_count=0) | 只拒 None/dict，核对数量（[p3_chain.py:132](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L132>)–142）；无顺序、dtype、shape、角色证明 | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| parsed PrimFunc | NS(params=(),body=object()) | 只查 hasattr（[p3_chain.py:143](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L143>)–145） | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| conformance | NS(verified=True,parsed_prim_func=同一个假对象,三组匹配 identity) | 属性读取 + is 同对象 + 字串匹配（[p3_chain.py:154](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L154>)–167），无 verifier 产物类型/来源 | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| artifact identity | 任意非空 source/tir/target/toolchain；协调 plan/tl | 前四项只查非空（[p3_chain.py:146](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L146>)–149），plan/tl 仅匹配 record（150–153） | [p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L201>) |
| entry / IR | 实际安装 PrimFunction，篡改 record entry/IR 字串 | 较硬：[p3_chain.py:69](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L69>) isinstance(PrimFunction)，[p3_chain.py:183](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L183>)–187 从实际 IR 重算并匹配 | ENTRY_IDENTITY_MISMATCH / STALE_IR_IDENTITY；仍有最后兜底 |

## 移除无条件拒绝前的最小加固清单（建议，未实施）

| 必须新增的校验 | 建议稳定拒绝码 |
|---|---|
| record 必须真实 ChainRecord；所有 identity 必须规范类型且来自实际交易，不信任 APPLIED 字段 | CHAIN_RECORD_NOT_ACTUAL / CHAIN_RECORD_UNVERIFIED |
| plan/completion/storage 必须是生产阶段定义的真实类型；重新计算内容身份、绑定同次交易和 iteration，并实际运行 verifier | PLAN_ARTIFACT_NOT_ACTUAL / COMPLETION_ARTIFACT_NOT_ACTUAL / STORAGE_ARTIFACT_NOT_ACTUAL；*_IDENTITY_MISMATCH / *_UNVERIFIED |
| 从 artifact.source 由真实 TL reader 解析，检查实际 TVM PrimFunc 类型并将 parser 产物与源哈希绑定 | TL_SOURCE_PARSE_FAILED / PARSED_PRIMFUNC_NOT_ACTUAL / SOURCE_IDENTITY_MISMATCH |
| ABI 由实际安装 entry 和实际 parsed TL 分别推导，并比较顺序/输入输出/shape/dtype/storage/scalar | ABI_NOT_ACTUAL / ABI_CONFORMANCE_MISMATCH |
| conformance 是实际 verifier 生成且绑定源、parsed、entry、plan 的真实结果；不可仅依赖 verified=True | TL_CONFORMANCE_NOT_ACTUAL / TL_CONFORMANCE_SOURCE_MISMATCH |
| 重算并比较 source、TIR、TL identity；绑定实际目标、编译器和工具链版本/配置 | ARTIFACT_TIR_IDENTITY_MISMATCH / ARTIFACT_TARGET_IDENTITY_MISMATCH / ARTIFACT_TOOLCHAIN_IDENTITY_MISMATCH |
| build/export 的发布边界必须强制调用 raising=True gate，拒绝 cached False verdict 或绕过 AFTER_LOWER | EXPORT_GATE_NOT_ENFORCED / EXPORT_NOT_RUNNABLE |
| 保留实际 PrimFunction/IR identity 校验并增加正向真实产物测试 + 假工件全套反向测试 | ENTRY_IDENTITY_MISMATCH / STALE_IR_IDENTITY（现有） |

类型校验只是必要条件；公共 dataclass 构造器仍可手造，不能替代内容重新推导和真实 verifier。

## 版本差异与旁路证据

只读 `git diff handoff-20261112 HEAD -- <六个必读实现文件>` exit 0：只有 p3_chain identity 实现变化，其余五个接口文件无差异。pre-f341e47 的 _describe 对非 dataclass Op 只保留 kind/name，身份可能碰撞；post-f341e47 使用 plan.identity。鸭子接缝和无条件拒绝逻辑在两个版本的 diff 中未变化；因此其源代码结构发现适用标签，**本脚本导入 plan.identity，标签上不能原样复跑**，标签动态结果是 UNKNOWN。

正常 package 入口 [__init__.py:6](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/__init__.py#L6>)–9 只导入 ir/target/pipeline/p3_chain。内容搜索 active target 中 `_archive_historical|attnres\.twin|mla\.kernel|materialize|tilelang`，没有历史导入或直接编译调用；直接 tilelang.compile 只有 builder docstring 提及。[test_native_builders.py:162](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_native_builders.py#L162>)–168 静态断言 builder 无直接 compile/twin/kernel。

真实 import 探针：attnres.twin、mla.kernel、completion.materialize、codegen.native_tl 均 ModuleNotFoundError；attnres.build 和 mla.build 可正常 import。进程最终 sys.modules 中 _archive_historical、tilelang、torch_npu 均为空。仅能支持“这些已知旧入口从当前正常 import 路径未到达”；不证明任意 sys.path 注入或显式加载归档文件不可能。归档文件 glob 未返回文件；历史实物是否存放在别处 UNKNOWN。未读取任何旧 worktree。

## 空模块、零函数、零测试

- Module('empty',()) 可构造，但 gate 无链时缺 CHAIN_NOT_RUN；伪 APPLIED 链时缺 ACTUAL_ENTRY_MISSING、三类 artifact missing、TL_ENTRY_MISSING；均 ready=False。
- pass_empty 返回成功但 metadata ready=False，不能将“无异常”作为导出成功。
- 以下两个命令实际执行：完整 suite exit 0，`24 passed in 0.29s`；零筛选 exit 5，`24 deselected in 0.28s`。exit 5 是期望拒绝，不是假通过。未发现 pytest 零测试会 exit 0；外部 gate 是否吞掉 exit 5 为 UNKNOWN（任务未提供执行门适配器）。


```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -p no:cacheprovider -q targets/ascend/tests_native_mainline
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -p no:cacheprovider -q targets/ascend/tests_native_mainline -k __challenge_zero_selected__
```

读取了三个顶层测试文件的全部内容（builders 177 行，identity 162 行，search_rank 84 行）；调查中另见 backend_operators 新文件并读取。24 passed 是命令当时结果，不保证并行团队继续提交后数量不变。

## 每条探针实际结果及可复制命令

58 条筛选命令也已各自启动独立 Python 进程实测，全部 exit=0、stderr 为空；退出码及原始 stdout 见 [command_exit_results.jsonl](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/probes/command_exit_results.jsonl>)。脚本捕获预期异常用于报告，**下列每个命令进程 exit code 均为 0**；不能把脚本 exit 0 当作 compile PASS。全量执行共 58 条 JSONL，未出现 expected_present=False，所有 gate ready=False。gate 异常顶层 code 为 EXPORT_NOT_RUNNABLE；细分拒绝码在 missing 中。MainlineNotImplemented 无 .code，表内 code 为实际 str(exception)。精确异常串均列在每条输出；不省略为“通过”。选择参数仅过滤输出，全量构造仍执行。

### no_chain

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py no_chain
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): chain:CHAIN_NOT_RUN, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "CHAIN_NOT_RUN",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "no_chain",
  "ready": false,
  "unchanged": true
}
```

### plan_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py plan_dict
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): plan:PLAN_ARTIFACT_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "PLAN_ARTIFACT_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "plan_dict",
  "ready": false,
  "unchanged": true
}
```

### plan_empty_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py plan_empty_object
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): plan:PLAN_ARTIFACT_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "PLAN_ARTIFACT_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "plan_empty_object",
  "ready": false,
  "unchanged": true
}
```

### plan_mismatch

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py plan_mismatch
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): plan:PLAN_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "PLAN_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "plan_mismatch",
  "ready": false,
  "unchanged": true
}
```

### plan_matching_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py plan_matching_object
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 1 gap(s): tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "missing": [
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "plan_matching_object",
  "ready": false,
  "unchanged": true
}
```

### completion_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py completion_dict
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): completion:COMPLETION_ARTIFACT_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "COMPLETION_ARTIFACT_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "completion_dict",
  "ready": false,
  "unchanged": true
}
```

### completion_empty_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py completion_empty_object
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): completion:COMPLETION_ARTIFACT_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "COMPLETION_ARTIFACT_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "completion_empty_object",
  "ready": false,
  "unchanged": true
}
```

### completion_mismatch

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py completion_mismatch
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): completion:COMPLETION_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "COMPLETION_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "completion_mismatch",
  "ready": false,
  "unchanged": true
}
```

### completion_matching_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py completion_matching_object
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 1 gap(s): tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "missing": [
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "completion_matching_object",
  "ready": false,
  "unchanged": true
}
```

### storage_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py storage_dict
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): storage:STORAGE_ARTIFACT_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "STORAGE_ARTIFACT_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "storage_dict",
  "ready": false,
  "unchanged": true
}
```

### storage_empty_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py storage_empty_object
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): storage:STORAGE_ARTIFACT_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "STORAGE_ARTIFACT_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "storage_empty_object",
  "ready": false,
  "unchanged": true
}
```

### storage_mismatch

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py storage_mismatch
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): storage:STORAGE_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "STORAGE_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "storage_mismatch",
  "ready": false,
  "unchanged": true
}
```

### storage_matching_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py storage_matching_object
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 1 gap(s): tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "missing": [
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "storage_matching_object",
  "ready": false,
  "unchanged": true
}
```

### native_tl_source_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_tl_source_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:TL_SOURCE_MISSING, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "TL_SOURCE_MISSING",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_tl_source_missing",
  "ready": false,
  "unchanged": true
}
```

### native_pass_configs_mismatch

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_pass_configs_mismatch
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:PASS_CONFIGS_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "PASS_CONFIGS_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_pass_configs_mismatch",
  "ready": false,
  "unchanged": true
}
```

### native_abi_not_actual

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_abi_not_actual
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:ABI_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ABI_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_abi_not_actual",
  "ready": false,
  "unchanged": true
}
```

### native_abi_input_count_mismatch

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_abi_input_count_mismatch
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:ABI_INPUT_COUNT_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ABI_INPUT_COUNT_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_abi_input_count_mismatch",
  "ready": false,
  "unchanged": true
}
```

### native_abi_output_count_mismatch

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_abi_output_count_mismatch
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:ABI_OUTPUT_COUNT_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ABI_OUTPUT_COUNT_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_abi_output_count_mismatch",
  "ready": false,
  "unchanged": true
}
```

### native_parsed_primfunc_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_parsed_primfunc_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 3 gap(s): tl:PARSED_PRIMFUNC_MISSING, tl:TL_CONFORMANCE_PARSED_PRIMFUNC_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "PARSED_PRIMFUNC_MISSING",
    "TL_CONFORMANCE_PARSED_PRIMFUNC_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_parsed_primfunc_missing",
  "ready": false,
  "unchanged": true
}
```

### native_artifact_source_identity_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_artifact_source_identity_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:ARTIFACT_SOURCE_IDENTITY_MISSING, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ARTIFACT_SOURCE_IDENTITY_MISSING",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_artifact_source_identity_missing",
  "ready": false,
  "unchanged": true
}
```

### native_artifact_plan_identity_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_artifact_plan_identity_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 3 gap(s): tl:ARTIFACT_PLAN_IDENTITY_MISSING, tl:ARTIFACT_PLAN_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ARTIFACT_PLAN_IDENTITY_MISSING",
    "ARTIFACT_PLAN_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_artifact_plan_identity_missing",
  "ready": false,
  "unchanged": true
}
```

### native_artifact_tir_identity_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_artifact_tir_identity_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:ARTIFACT_TIR_IDENTITY_MISSING, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ARTIFACT_TIR_IDENTITY_MISSING",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_artifact_tir_identity_missing",
  "ready": false,
  "unchanged": true
}
```

### native_artifact_tl_identity_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_artifact_tl_identity_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 3 gap(s): tl:ARTIFACT_TL_IDENTITY_MISSING, tl:ARTIFACT_TL_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ARTIFACT_TL_IDENTITY_MISSING",
    "ARTIFACT_TL_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_artifact_tl_identity_missing",
  "ready": false,
  "unchanged": true
}
```

### native_artifact_target_identity_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_artifact_target_identity_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:ARTIFACT_TARGET_IDENTITY_MISSING, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ARTIFACT_TARGET_IDENTITY_MISSING",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_artifact_target_identity_missing",
  "ready": false,
  "unchanged": true
}
```

### native_artifact_toolchain_identity_missing

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_artifact_toolchain_identity_missing
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:ARTIFACT_TOOLCHAIN_IDENTITY_MISSING, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ARTIFACT_TOOLCHAIN_IDENTITY_MISSING",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_artifact_toolchain_identity_missing",
  "ready": false,
  "unchanged": true
}
```

### native_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py native_dict
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:TL_ARTIFACT_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "TL_ARTIFACT_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "native_dict",
  "ready": false,
  "unchanged": true
}
```

### conformance_none

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py conformance_none
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:TL_CONFORMANCE_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "TL_CONFORMANCE_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "conformance_none",
  "ready": false,
  "unchanged": true
}
```

### conformance_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py conformance_dict
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:TL_CONFORMANCE_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "TL_CONFORMANCE_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "conformance_dict",
  "ready": false,
  "unchanged": true
}
```

### conformance_verified_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py conformance_verified_dict
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): tl:TL_CONFORMANCE_NOT_ACTUAL, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "TL_CONFORMANCE_NOT_ACTUAL",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "conformance_verified_dict",
  "ready": false,
  "unchanged": true
}
```

### conformance_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py conformance_object
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 5 gap(s): tl:TL_CONFORMANCE_PARSED_PRIMFUNC_MISMATCH, tl:TL_CONFORMANCE_ENTRY_IDENTITY_MISMATCH, tl:TL_CONFORMANCE_PLAN_IDENTITY_MISMATCH, tl:TL_CONFORMANCE_TL_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "TL_CONFORMANCE_PARSED_PRIMFUNC_MISMATCH",
    "TL_CONFORMANCE_ENTRY_IDENTITY_MISMATCH",
    "TL_CONFORMANCE_PLAN_IDENTITY_MISMATCH",
    "TL_CONFORMANCE_TL_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "conformance_object",
  "ready": false,
  "unchanged": true
}
```

### tamper_ir_identity

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py tamper_ir_identity
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): chain:STALE_IR_IDENTITY, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "STALE_IR_IDENTITY",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "tamper_ir_identity",
  "ready": false,
  "unchanged": true
}
```

### tamper_entry_identity

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py tamper_entry_identity
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 3 gap(s): entry:ENTRY_IDENTITY_MISMATCH, tl:TL_CONFORMANCE_ENTRY_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ENTRY_IDENTITY_MISMATCH",
    "TL_CONFORMANCE_ENTRY_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "tamper_entry_identity",
  "ready": false,
  "unchanged": true
}
```

### all_structural_fields_forged

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py all_structural_fields_forged
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 1 gap(s): tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "all_structural_fields_forged",
  "ready": false,
  "unchanged": true
}
```

### chain_duck_record

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py chain_duck_record
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 1 gap(s): tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "chain_duck_record",
  "ready": false,
  "unchanged": true
}
```

### coordinated_invented_identities

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py coordinated_invented_identities
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 1 gap(s): tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "coordinated_invented_identities",
  "ready": false,
  "unchanged": true
}
```

### record_identity_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py record_identity_object
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 4 gap(s): plan:PLAN_IDENTITY_MISMATCH, tl:ARTIFACT_PLAN_IDENTITY_MISMATCH, tl:TL_CONFORMANCE_PLAN_IDENTITY_MISMATCH, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "PLAN_IDENTITY_MISMATCH",
    "ARTIFACT_PLAN_IDENTITY_MISMATCH",
    "TL_CONFORMANCE_PLAN_IDENTITY_MISMATCH",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "record_identity_object",
  "ready": false,
  "unchanged": true
}
```

### chain_record_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py chain_record_dict
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": null,
  "detail": "'dict' object has no attribute 'entry_identity'",
  "exception": "AttributeError",
  "probe": "chain_record_dict"
}
```

### install_none

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py install_none
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "INSTALL_CANDIDATE_UNVERIFIED",
  "detail": "INSTALL_CANDIDATE_UNVERIFIED: install_entry requires the actual verify_actual_program conformance result; a same-name candidate and caller metadata are not verification",
  "exception": "PlanError",
  "fresh_candidate": true,
  "probe": "install_none",
  "unchanged": true
}
```

### install_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py install_dict
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "INSTALL_CANDIDATE_UNVERIFIED",
  "detail": "INSTALL_CANDIDATE_UNVERIFIED: install_entry refuses a self-reported conformance dict; actual verification is required",
  "exception": "PlanError",
  "fresh_candidate": true,
  "probe": "install_dict",
  "unchanged": true
}
```

### install_verified_dict

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py install_verified_dict
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "INSTALL_CANDIDATE_UNVERIFIED",
  "detail": "INSTALL_CANDIDATE_UNVERIFIED: install_entry refuses a self-reported conformance dict; actual verification is required",
  "exception": "PlanError",
  "fresh_candidate": true,
  "probe": "install_verified_dict",
  "unchanged": true
}
```

### install_object

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py install_object
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "INSTALL_VERIFIER_NOT_IMPLEMENTED",
  "detail": "INSTALL_VERIFIER_NOT_IMPLEMENTED",
  "exception": "MainlineNotImplemented",
  "fresh_candidate": true,
  "probe": "install_object",
  "unchanged": true
}
```

### compile_mainline

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py compile_mainline
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "MAINLINE_CHAIN_NOT_IMPLEMENTED",
  "detail": "MAINLINE_CHAIN_NOT_IMPLEMENTED",
  "exception": "MainlineNotImplemented",
  "probe": "compile_mainline",
  "unchanged": true
}
```

### apply_symbolic_pipeline

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py apply_symbolic_pipeline
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "MAINLINE_APPLY_NOT_IMPLEMENTED",
  "detail": "MAINLINE_APPLY_NOT_IMPLEMENTED",
  "exception": "MainlineNotImplemented",
  "probe": "apply_symbolic_pipeline",
  "unchanged": true
}
```

### verify_actual_program

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py verify_actual_program
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "MAINLINE_VERIFY_NOT_IMPLEMENTED",
  "detail": "MAINLINE_VERIFY_NOT_IMPLEMENTED",
  "exception": "MainlineNotImplemented",
  "probe": "verify_actual_program",
  "unchanged": true
}
```

### emit_supported_tl

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py emit_supported_tl
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "TL_EMIT_NOT_IMPLEMENTED",
  "detail": "TL_EMIT_NOT_IMPLEMENTED",
  "exception": "MainlineNotImplemented",
  "probe": "emit_supported_tl",
  "unchanged": true
}
```

### solve_region

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py solve_region
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "SOLVER_SHARED_REGION_NOT_IMPLEMENTED",
  "detail": "SOLVER_SHARED_REGION_NOT_IMPLEMENTED",
  "exception": "MainlineNotImplemented",
  "probe": "solve_region",
  "unchanged": true
}
```

### place_resolved_versions

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py place_resolved_versions
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "code": "STORAGE_PLACEMENT_NOT_IMPLEMENTED",
  "detail": "STORAGE_PLACEMENT_NOT_IMPLEMENTED",
  "exception": "MainlineNotImplemented",
  "probe": "place_resolved_versions",
  "unchanged": true
}
```

### empty_no_chain

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py empty_no_chain
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 2 gap(s): chain:CHAIN_NOT_RUN, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "CHAIN_NOT_RUN",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "empty_no_chain",
  "ready": false,
  "unchanged": true
}
```

### empty_forged_chain

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py empty_forged_chain
```

实际进程 exit=0；预期拒绝码命中。

```json
{
  "code": "EXPORT_NOT_RUNNABLE",
  "detail": "EXPORT_NOT_RUNNABLE: a runnable artifact requires the actual installed entry and real parsed TL/ABI conformance; 6 gap(s): entry:ACTUAL_ENTRY_MISSING, plan:PLAN_ARTIFACT_MISSING, completion:COMPLETION_ARTIFACT_MISSING, storage:STORAGE_ARTIFACT_MISSING, tl:TL_ENTRY_MISSING, tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED",
  "exception": "ExportRefusal",
  "expected_present": true,
  "missing": [
    "ACTUAL_ENTRY_MISSING",
    "PLAN_ARTIFACT_MISSING",
    "COMPLETION_ARTIFACT_MISSING",
    "STORAGE_ARTIFACT_MISSING",
    "TL_ENTRY_MISSING",
    "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"
  ],
  "probe": "empty_forged_chain",
  "ready": false,
  "unchanged": true
}
```

### pass_minimal

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py pass_minimal
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "ir_unchanged": true,
  "metadata_keys": [
    "ascend.export_gate",
    "ascend.native_mainline_chain"
  ],
  "probe": "pass_minimal",
  "ready": false,
  "reason": "MAINLINE_CHAIN_NOT_IMPLEMENTED",
  "same_module": true
}
```

### pass_empty

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py pass_empty
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "ir_unchanged": true,
  "metadata_keys": [
    "ascend.export_gate",
    "ascend.native_mainline_chain"
  ],
  "probe": "pass_empty",
  "ready": false,
  "reason": "MAINLINE_CHAIN_NOT_IMPLEMENTED",
  "same_module": true
}
```

### import_tilefoundry_ascend.attnres.twin

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py import_tilefoundry_ascend.attnres.twin
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "detail": "No module named 'tilefoundry_ascend.attnres.twin'",
  "exception": "ModuleNotFoundError",
  "probe": "import_tilefoundry_ascend.attnres.twin"
}
```

### import_tilefoundry_ascend.mla.kernel

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py import_tilefoundry_ascend.mla.kernel
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "detail": "No module named 'tilefoundry_ascend.mla.kernel'",
  "exception": "ModuleNotFoundError",
  "probe": "import_tilefoundry_ascend.mla.kernel"
}
```

### import_tilefoundry_ascend.completion.materialize

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py import_tilefoundry_ascend.completion.materialize
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "detail": "No module named 'tilefoundry_ascend.completion.materialize'",
  "exception": "ModuleNotFoundError",
  "probe": "import_tilefoundry_ascend.completion.materialize"
}
```

### import_tilefoundry_ascend.codegen.native_tl

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py import_tilefoundry_ascend.codegen.native_tl
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "detail": "No module named 'tilefoundry_ascend.codegen.native_tl'",
  "exception": "ModuleNotFoundError",
  "probe": "import_tilefoundry_ascend.codegen.native_tl"
}
```

### import_tilefoundry_ascend.attnres.build

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py import_tilefoundry_ascend.attnres.build
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "exception": null,
  "file": "/home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/attnres/build.py",
  "probe": "import_tilefoundry_ascend.attnres.build"
}
```

### import_tilefoundry_ascend.mla.build

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py import_tilefoundry_ascend.mla.build
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "exception": null,
  "file": "/home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/mla/build.py",
  "probe": "import_tilefoundry_ascend.mla.build"
}
```

### loaded_bypass_modules

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/challenge_probes.py loaded_bypass_modules
```

实际进程 exit=0；观察项（局部放行/导入/阶段行为，见结论）。

```json
{
  "modules": [],
  "probe": "loaded_bypass_modules"
}
```

## 可转成正式 active 测试（18 组建议）

1. 无 CHAIN_KEY：断言 CHAIN_NOT_RUN + ready=False，并核对 ExportRefusal.missing。
2. 三类 metadata dict：参数化断言 *_ARTIFACT_NOT_ACTUAL。
3. 三类 metadata 无 identity：参数化拒绝。
4. 三类身份不匹配：参数化 *_IDENTITY_MISMATCH。
5. 三类匹配 identity 的鸭子对象：加固后必须拒绝，当前是已证实盲区。
6. 非 ChainRecord 属性齐全对象与 dict：加固后统一 CHAIN_RECORD_NOT_ACTUAL，不能 AttributeError。
7. NativeArtifact source 空/非字符串：TL_SOURCE_MISSING。
8. 单键、错值或错序 PassConfigs：PASS_CONFIGS_MISMATCH。
9. ABI dict/假 object：加固后 ABI_NOT_ACTUAL；真实 ABI 分别错输入和输出。
10. parsed 缺失/假 params+body 对象：分别现有码与建议 PARSED_PRIMFUNC_NOT_ACTUAL。
11. 六个 artifact identity 空值参数化；plan/tl 另核对 mismatch。
12. source/tir/target/toolchain identity 非空但错误：加固后重算拒绝。
13. conformance None/dict/verified dict 均拒绝。
14. conformance verified=False、parsed 非同对象和三类 identity 错配分别拒绝；完整鸭子对象加固后也拒绝。
15. 同名 fresh PrimFunction 安装：None/dict/object conformance 拒绝且 Module 未变。
16. record.ir_identity/entry_identity tamper 各命中稳定拒绝。
17. 六个直接阶段 MainlineNotImplemented，核对 IR 与 metadata 不变；pass.run 单列 metadata 变化契约。
18. 空 Module 不能 export；suite 零收集 exit 非 0；真实有效工件正向测试不得以空集合/全 skip 替代。

现有 stub 位置：[mainline.py:75](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py#L75>)、78–87；[__init__.py:31](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/codegen/__init__.py#L31>)；[__init__.py:26](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/solver/__init__.py#L26>)；[__init__.py:30](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/__init__.py#L30>)。CompletionPlan/StoragePlan 只是公开可构造 dataclass，见 [__init__.py:20](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/completion/__init__.py#L20>)、[__init__.py:22](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/__init__.py#L22>)。探针只报告进程内 IR/metadata/参数可见状态，不将未监控的外部状态称为证明无副作用。
