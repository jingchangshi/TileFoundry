# 07 实施接口、代码迁入与产物合同

## 7.1 唯一入口

正式路径保留现有 `checked_program`、`analyze`、`schedule.finalize` 与 `emit_for_codegen`。在现有 `RunAscendAnalysisChain` 中替换错误投影/静态物化。新增 build/operator worker 是该入口的调用者，不是另一条编译语义。

`attnres/build.py`、`mla/build.py` 应返回 native Module。runtime 轻入口只读取已完成 artifact 及 ABI。它不得从 case JSON 重新生成调度计划。

## 7.2 核心接口与负责人

| 接口 | 输入 | 输出 | owner |
|---|---|---|---|
| normalize_intent | native HIR、target | 原生 intent、region binding 摘要 | HIR/analysis |
| form_schedule_units | target 合法 TIR、source map | native UnitTree | HIR/analysis |
| analyze_schedule | units、effects、alias、cost | 有限 region problem 及边界关系 | HIR/analysis |
| solve_region | typed problem, cost, remaining deadline | SAT/UNSAT/UNKNOWN + assignment | scheduler |
| resolve_schedule | 原 source/intent、assignment | 唯一 ResolvedPlan | scheduler |
| build_symbolic_completion | plan、真实 effects、target 机制 | 参数化 channels/sites | completion |
| place_resolved_storage | plan、completion、physical layouts | 正式 StoragePlan 或具体 replan cut | storage/backend |
| apply_symbolic_pipeline | 原 entry、完整 plan、whole-task refs | 实际新 entry 与 source map | compiler |
| verify_actual_ir | 源、plan、实际 entry | 语义覆盖、slot/guard/SSA/资源结论 | verifier |
| emit_supported_tl | 实际 entry、ABI、target registry | 显式 TL 和固定 PassConfig | backend |
| read_actual_tl | 实际解析 PrimFunc、template registry | 可比较语义事实 | verifier/backend |
| bind/invoke | 冻结 artifact、输入 tensor/metadata | 单次调用及输出 | runtime |

接口名称可在 P1 依项目风格调整一次。调整应同步调用者、测试、规格和启动计划；不得出现第二套同职责 authority。

## 7.3 本包代码的迁入位置

| 包内实现 | 拟迁入/替换位置 | 已完成范围 | 本地仍需完成 |
|---|---|---|---|
| `implementation/modulo_kernel.py` | shared vendor core/adapter + `solver/bridge.py` | 真 Z3 固定 II 约束与独立赋值检查 | native 抽取/coverage、成本/owner 适配、统一旧 API/provenance |
| `implementation/symbolic_template.py` | `plan/pipeline.py` | O(tasks）符号循环构造、短 loop 公式、dense FREE 边界 | 原生 Builder、task 克隆/stream/guard、completion 挂接 |
| `implementation/native_bindings.py` | `plan/pipeline.py`/已有 visitor helper | 真实 TF 类型的 binder 克隆与 entry 安装代码；语法/API 审查 | 实际 TF 测试、type/effect/mesh 重写、纯 scalar 支持 |
| `implementation/compile_transaction.py` | `passes/p3_chain.py` 的 RunAscendAnalysisChain | 共同 deadline、有限 replan、实际 entry 检查、失败不导出 | 各 hook 接原生服务、最终 semantic identity 与后端读取 |
| `reference_core/*.py` | 算子/协议独立测试 oracle | online 数学、AR literal/factored、MLA canonical/deferred/列片、epoch 反例 | 不作为 production compiler 或设备实现 |
| `tools/run_contract.py` | 正式 runtime 验证调用层或保留工具 | 调原 test.main、完整 key 与 fixture 身份、日志 | 实际比较器/CANN/设备、单 launch 与性能另测 |

本包没有宣称这些 helper 已经合入 main。不要把 35 项框架测试通过写成 TF pass 通过；不要把 20 项数学测试写成原 40 例通过。native 接入应运行项目的真实类型检查、printer 和设备链。

## 7.4 ApplyPipeline 的 hook 不是空白授权

`instantiate_whole_task` 的实现须以 real unit 和 storage source map 为输入，逐原节点保留语义；不能在 hook 中硬编码一个 AR/MLA 全 kernel。`rewrite_type` 使用原有 type/layout substitution；`rewrite_effect` 只改明确需要的 generation/address 属性；`rewrite_mesh` 保留实际 owner 与维度。缺少必要能力时实现它，不用 mock 在正常入口通过。

包中 native 克隆器为受限 TIR 表达式白名单。任务内 stream allocation 应先 hoist；带独立 PipelineSpec 的 nested loop 先由对应 region 处理。遇到 unsupported 应有 source 定位，不能偷偷回退到按 name/shape 替换。

## 7.5 冻结与语义 identity

同一进程里，plan 与 native 源关联；跨进程使用正式 printer 产物、受支持 TL 源码和 artifact descriptor。一个语义 identity 应覆盖：operation 及属性、全部 operand 与 binder、常数/精度、type/layout/offset、For/If/guard、core/owner、task 多重性、slot/version、completion 端点与实际顺序。

hash 是防串文件/变更失效手段，不是等价证明。source→TIR 应按约定 substitution 验证；TIR→TL 应读取 actual PrimFunc 的支持模板。修改 Vmul 第二输入后，不能只因操作类型仍是 Vmul 而通过。

TL 关闭 AutoSchedule 后仍有布局、ND→NZ、UB 合并与 flag 重写；除 parsed PrimFunc 之外，还须按 04 节登记的 refinement 合同观察后期实际 TL IR/CCE 的资源和同步变化。首 profile 设置 `tl.disable_shared_memory_reuse=True`，允许 arena 重定位但禁独立 allocation 的生命周期复用；首版不接受未登记的 flag→mutex 溢出。普通 VF 合法降指令不能因 op 数量变化而误判。

冻结不能只是保存可被原对象联动修改的浅引用。生成 canonical semantic snapshot 或不可变 copy 后绑定 identity；导出前读取当前 actual entry 重核。缓存 key 含 plan/shape/dtype/layout/completion/target/scratch/toolchain 真实版本。

## 7.6 Artifact descriptor

推荐一个版本化 descriptor，不传完整 Python IR pickle：

```json
{
  "schema": "ascend.native-artifact/1",
  "source_commit": "actual commit",
  "source_identity": "actual semantic source hash",
  "plan_identity": "actual resolved plan hash",
  "tir_identity": "actual installed entry hash",
  "tl_source_path": "/absolute/path/kernel.py",
  "tl_source_sha256": "computed",
  "entry_symbol": "actual symbol",
  "abi": {"ordered_arguments": [], "output_indices": []},
  "target_profile": "actual pinned profile",
  "tilelang_build": "actual version",
  "pass_configs": {"tl.enable_auto_schedule": false, "tl.disable_shared_memory_reuse": true},
  "verification": {"native_ir": "PASS", "tl_mapping": "PASS"}
}
```

这是格式示意，不能把示意字段值作为真实成功产物。预编译时未知 binary identity 可以留空并标未观察；真正 compile/load 之后补实际信息。ABI 应包括输入/scalar 顺序、shape/dtype/stride、owner 无关参数、输出合同。runtime 只校验 metadata 并分配输出；语义由已编译 kernel 完成。

## 7.7 任务集成规则

每个 agent 持独立工作树。lead 拥有 `resolved_plan` 公共 schema、PassManager 注册、主集成入口和版本选择；其他 owner 提交小型 patch 并在共享接口冻结后接线。避免多个 agent 同时替换同一核心文件。

任务完成需包含改动原因、实际入口、测试命令、结果和边界。一个模块的 CPU 结果不能替另一个模块填 PASS。每阶段先形成一条真实纵向切片，再增加支持域和性能；最终所有算子都回到同一入口。
