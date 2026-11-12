# S0-entry 报告：真实作者样例经正常入口到下界

## 改动（只新增，未修改任何既有文件）
- targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py：真实作者源码。@module(entry="vsum", topologies=(Topology("aiv",1),)) + @func；一个带 tf.pipelined(4, max_stage=2, max_buffer_versions=2) 的循环，循环体在 Mesh(("aiv",)) 内的 T.SimdVF 里做 GM→UB→vld/pset/vadd/vsts→UB 的真实 64-lane 累加（base[k+1]=base[k]+x），函数返回尾部 GM tensor。
- targets/ascend/tests_native_mainline/test_normal_entry.py：5 项测试；4 项走 checked_program → analyze(analysis=("compute-cost","memory")) → finalize，1 项检查解析后的 typed pipeline 声明。
- targets/ascend/tilefoundry_ascend/ir/**：未改。现有 SIMD_VF 注册（ir/execution.py:43-65）与 AIV 指令注册（ir/ops.py）已能承接该写法，没有缺口需要补；未动 passes/plan/codegen。

## 命令与实际输出
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests_native_mainline
→ 29 passed in 0.89s（原有 24 + 新增 5；无 skip/xfail）。

finalize 后实测（同一入口，未手搓 TIR）：
- entry = PrimFunction "vsum"，params=[x, out]，output_count=1；ABI output_split=1，inputs=["x"]，outputs=["out"]（role=tensor, storage=gmem, shape=(64,)）。
- TIR：1 个 For，其 pipeline 与作者 PipelineSpec 相同（max_stage=2, max_buffer_versions=2, enable_offset=False）；1 个 ExecutionScope，kind="ascend.simdvf"；指令计数 GmToUb=2、Vld=2、Pset=1、Vadd=1、Vsts=1、UbToGm=1（循环体 1 次 GM 取片、5 条域内指令、循环外初载与末写）。
- ascend.native_mainline_chain = ChainRecord(stopped_at="freeze", reason="MAINLINE_CHAIN_NOT_IMPLEMENTED", pipeline_status="REFUSED", applied=(), ir_identity 与当前 module 一致)。是具名阶段记录，不是异常逃逸。
- ascend.export_gate：ready=False，含具名 gaps：chain:MAINLINE_CHAIN_NOT_IMPLEMENTED、chain:TRANSACTION_INCOMPLETE、chain:CHAIN_STOPPED_AT_FREEZE、entry:ENTRY_IDENTITY_MISMATCH、plan:PLAN_ARTIFACT_MISSING、tl:NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED 等。
- printer：输出含 tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=False) 与 with T.SimdVF():；写盘重载后 as_script 与原文逐字相同。

## 复审后加强（reviewer-g1 建议）
- export gate 不只读缓存：同一 module 上 require_export_ready(raising=False) 重算结果与缓存 gate 相等；raising=True 抛 ExportRefusal("EXPORT_NOT_RUNNABLE")，gate="entry" 且 missing 含预期具名 codes。
- 末次 UbToGm 的目的对象不是按名字查：断言 stores[0].args[-1] is entry.params[-1]（对象 identity），写错目标就红。

## 为什么这样写
合法作者组合 = Mesh 包住 SimdVF；SimdVF 内只发 inside 指令（vld/pset/vadd/vsts），GM 传输在域外。legacy 二元运算/泛型 copy 在 SimdVF 内会被 verify 具名拒绝（visitor_registry/verify.py:1205），那是设计内的拒绝，未放宽、未绕 ConvertHIRToTIR。

## 剩余缺口
- 链在 compile_mainline 处 fail-closed；solver/plan/completion/storage/TL/install/export 全部仍未实现，export 恒 not ready。这是 S0-entry 入口证据，不是 S0 完成。
- S0 交付条件（安装后实际 TIR 两组输入 CPU evaluator、TL operand/ABI/PassConfig 对应、改线使认证失效）未做；本任务明确不做 solver/plan/emit。
- 作者样例只有一份；§1.4 的第二份 MIX/AIC-AIV transfer 样例不在本任务范围。
- pipeline 仅 D0 声明：无 task_intents/buffer_schedules，无 stage/II/V 量化。
