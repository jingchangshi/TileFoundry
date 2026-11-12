# mainline_cleanup_manifest.md 草案

状态：CLEAN-ARCH 只读清单；未执行移动、删除、生产编辑或测试。审计快照：本会话 TileFoundry_native_mainline，分支 feat/ascend950pr-native-mainline-local，HEAD ef69b4f；开始时 git status --short 为空。下列路径均相对会话 workspace；T=TileFoundry_native_mainline，A=artifacts/native_v3，P=T/targets/ascend/tilefoundry_ascend。本文交给 CLEAN-DOCS 落盘为 A/research/mainline_cleanup_manifest.md；它是当前清理清单，不能随旧 research 整目录归档。

## KEEP_ACTIVE
| 路径 | 理由 | 建议动作 |
|---|---|---|
| A/DECISION_NATIVE_MAINLINE_ONLY.md | 当前单链决策，无历史生产兼容 | 保留；更新旧 Agent 名与写者归属；移除“先读旧研究报告”的任务前提 |
| A/INTERFACE_FREEZE_NATIVE_MAINLINE.md | 当前 schema/API/事务合同 | 保留；落实 task-105：唯一 writer、effects 参数、deadline 单位、source_map 返回、TL conformance/commit 阶段门 |
| A/CHECKPOINT_MAINLINE_RESET.md | 唯一当前工作树/分支/状态锚点 | 保留；补 CLEAN-107/108/109 阻塞关系；旧116/1仅基线、不能沿用为新门 |
| A/ACTIVE_ONLY.md、A/ACTIVE_CONTRACT.md、T/ACTIVE_CONTRACT.md（待建） | 活动入口需要白名单 | 写当前唯一工作树、主链、禁止归档作实现依据、原40数值合同及阶段状态；彼此链接，不复制一套不同接口 |
| A/当前状态文件（由Lead明确唯一名称） | 当前 PASS/NOT_RUN 必须有来源 | 只保留本轮状态；旧 progress 归档，新状态不能继承旧 CPU PASS |
| P/search/process.py 的 WorkerCommand/Deadline/run_process | 已有独立进程/超时机制，不是版本兼容层 | 复用；不重新造进程框架 |
| 原 fixtures/golden/comparator/thresholds、真实依赖/同步/容量/ABI/identity/失败事务检查 | 正确性合同 | 保留；不得因“防御”二字删除。UNKNOWN容量拒绝、未知op具名拒绝是主线要求 |

## ARCHIVE_DO_NOT_FOLLOW
| 路径 | 理由 | 建议动作 |
|---|---|---|
| A/checkpoint.md、CHECKPOINT_LEAD.md、PLAN_BREAKDOWN.md、commands.md | 指向 ../TileFoundry_native_v3、旧WS/Agent/阶段门与静态回归 | 原字节移到 A/_archive_compat_defensive/，保留相对层级/hash；加归档索引，不改原报告 |
| A/research/completion_storage_design.md、completion_storage_test_vectors.md、pipeline_migration_checklist.md、storage_migration_checklist.md | 旧commit/旧任务/API迁移草案，不是现冻结契约 | 同上；可用事实若要重用，应经当前源复核后写入活动合同，不能引用旧任务写范围 |
| A/research/emitter_tl_reader_checklist.md、runtime_worker_checklist.md | 旧文明确保留静态micro入口/task40；emitter第36行要求保留static micro regression，与新决策冲突 | 全文归档；从task-99“先读”中去掉；当前六模板/双worker要求直接取当前合同 |
| A/research/ar_builder_checklist.md、mla_builder_checklist.md、target_facts_inventory.md | 数值/部署事实有价值，但快照路径/旧writer/API已过时，不能整体作为活动权威 | 全文归档；原40/golden不动；硬件事实只按当前Target身份复核后采纳，不重标DEVICE_VALIDATED |
| A/validation/ 全部旧报告与配套JSON/log/probe，A/baseline/、旧 *regression*.log / reproduce / transaction / semantic / pipeline-v1 日志，旧environment/progress记录（若存在） | 固定历史commit、旧数值/模型/环境证据，不证明新mainline | 保留原始内容与sha归档；不要只移md留下JSON顶层PASS误导；新证据另用本轮唯一目录 |
| T/targets/ascend/REVIEW_HANDOFF.md、NATIVE_V3_STATUS.md、旧evidence/目录 | 仍称用户暂停、旧4x64支持域/旧缺陷/旧分支状态 | 由CLEAN-CODE移入 T/targets/ascend/_archive_historical/ 并写历史索引；活动入口改指ACTIVE_CONTRACT；原日志不改写 |
| ../TileFoundry_native_v3 与分支 feat/ascend950pr-native-v3；../TileFoundry_native_mainline（无-local的另一个旧树）；../TileFoundry_p3_02；../TileFoundry_task71_validation、task80_validation、task85_validation、task92_validation | git worktree list 实际仍注册；同HEAD不能等同活动树 | 历史登记/HISTORICAL_WORKTREE_DO_NOT_USE.md（Lead/获分配写者执行）；不删除、不reset、不clean、不改分支历史。不要把 ../TileFoundry 或 ../TileFoundry_main_review 擅自标废弃，它们是基线/主线审阅树 |
| 历史任务1..97中已完成记录，旧验证/修补任务 | 旧工作结果仅历史，不定义当前实施优先级 | 保留完成历史；若有旧pending应标superseded或从活动DAG摘除，不重新执行旧兼容验收；当前以98..109为准 |

## DELETE_OR_REWRITE_IN_MAINLINE
| 路径/锚点 | 理由 | 建议动作 |
|---|---|---|
| T/src/tilefoundry/ir/hir/pipeline.py:175-214、legacy_num_stages_mapping；parser/pattern_nodes.py:5163-5249；parser/pipeline_intents.py；inspection/python_printer.py:1240、tir_printer.py:279；analysis/pipeline_reader.py及cli/source.py相关调用 | v1/v2双schema、num_stages/annotations兼容surface、缺schema默认pipeline/1 | 改单一typed PipelineSpec，统一parser/reader/printer/CLI消费者；删版本分派和legacy mapping，不加adapter |
| T/src/tilefoundry/vendor/tilelang_z3/core.py:737-744、_validate_loop_inputs默认；P/solver/bridge.py:528、575、796及fallback_candidate/verify_fallback_candidate旧协议 | num_stages同时默认stage与V；遗留参数/serial fallback/UNKNOWN旧候选桥 | 删历史参数及默认映射，显式独立max_stage和V域；状态统一SAT/UNSAT/UNKNOWN。UNKNOWN不得借候选进入安装；保留真实求解硬约束/后验核查 |
| P/passes/p3_chain.py:364、473、515、937-942、987-999、1060、1085、1239以后；P/plan/materialize.py、semantic_correspondence.py静态range(trip)路径；plan/scheduled_verify.py __k命名计数 | static materialize/reemit_into_unrolled，漏task/access后source_tasks/unsupported_tasks补账，固定stage0/V1/f32形状 | 重写为完整NormalizedProblem→唯一ResolvedPlan→symbolic whole-task实际entry；无法建模时在构造阶段拒绝；移除静态生产路径和假覆盖，保留逐operand/SSA/guard独立语义核查能力 |
| P/completion/plan.py:255 reemit_into_unrolled、completion/emit.py 的unrolled trace routing、completion/schedule.py旧trace合同及completion/__init__.py导出；P/sync.py 对materialize依赖 | 按N展开与trace重发仍可用 | 删除旧入口、导出和调用，有限symbolic site/channel为唯一合同；迁移前追完真实引用，不能只藏在diagnostic_only开关后 |
| P/codegen/twin.py、attnres/twin.py、mixed/twin.py、schedule/resolved.py及attnres/__init__.py:96、attnres/model.py:358的导出/调用 | 即便默认拒绝，仍有explicit diagnostic_only与第二套plan/text API | 从活动生产包移除旧实现及导出，历史放归档；不保留兼容wrapper。真实ResolvedPlan与parsed PrimFunc reader唯一权威 |
| P/mla/kernel.py:403-409及AR旧kernel/hir业务入口 | 自建手写TL、绕TF计划直接compile；AR旧HIR不能作为第二权威 | native builder返回真实Module/For/ExecutionScope并走主链；旧数学策略可在native中正确重建，禁止留旁路或host数学 |
| P/codegen/native_aiv.py:46、78-81、118-126；codegen/x_runtime.py:396-404、422、438-453 | 单键PassConfig、4x64/N4/V1白名单、binary OBSERVED无条件拒绝、runtime占位 | 删除旧静态profile，固定两键实际编译策略；binary按当前capability记录；实现artifact/ABI bind/invoke。注意 backend内tilelang.compile本身必须保留，删除的是算子旁路，不是底层编译调用 |
| P/storage/placement.py:137-143及plan_placement/verify、P/schedule/ledger.py:1590等默认元数据路径 | legacy capacity_bytes=None仍fits=True、猜dtype/shape/access default | 删除旧构造器行为，真实Target/owner/layout元数据为准；UNKNOWN_CAPACITY拒绝，不以footprint回填或默认f32伪造 |
| T/targets/ascend/tests/test_native_aiv.py:36、native_aiv_device.py:52、test_plan_r0_r2.py/r3.py、test_resolve_schedule_p3.py、test_project_loop_problem_v3.py、test_l3_budget_versions.py、test_d0_pipeline_controls.py、test_pipeline_g32.py、旧g33/g40/g41/l4/l5测试族 | 固化单键、num_stages、静态展开、手工problem/plan、legacy schema身份 | 删过时断言或重写到checked_program→analyze→finalize；按行为审查，不按文件名批量删除所有正确性测试。拆出仍有效的失败事务/operand/slot/generation/owner/sync/容量断言 |
| tests/test_symbolic_entry_v3.py:58-60、89-91；test_unit_tree_step1.py:261；solver_contract_v3/solver_bridge_s0_s2/solver_bridge_s3/solver_p3_01_budget_stage的skipif；test_mla_g63.py:34/41/54 | CPU数值占位xfail、环境开关隐藏正常入口、缺依赖/fixture时skip造成假绿 | 删除占位，正常入口和两不同输入CPU真数值测试必须执行；正式验收缺依赖/fixture明确失败/环境阻断，不计通过；保留真实设备未跑状态 |
| tests/test_legacy_twin_v3.py、test_legacy_ar_mix_v3.py、test_mixed_twin_g43.py、test_attnres_g50.py中的legacy wrapper；其他旧导出测试 | 以兼容诊断API作为持续回归目标 | 删除旧API资格/诊断匹配合同；需要的“旧输入不能进入正常链”改为主线入口的简短拒绝测试，不维护旧实现 |

## 执行顺序与范围
1. CLEAN-DOCS 与 CLEAN-CODE均 blocked_by task-107，完成清单后可并行；MAIN-A/B均继续 blocked_by task-108/task-109。
2. Lead需将CLEAN-CODE scope补齐 inspection/*、analysis/pipeline_reader.py、plan/、completion/、storage/、sync.py、schedule/、mixed/及旧文档/evidence；当前task-109 scopes未覆盖上述实物。由单一实现者完成，MAIN-A/B保持冻结，不能各自修同一文件。
3. CLEAN-DOCS移动research前协调task-104（正在写artifacts/native_v3），并改task-99对已归档旧清单的强制读取要求；保留只读工作包20261112/tf_ascend950pr_native_v3及reference原样。
4. 文档移动记录source→destination→sha256，不能改写历史PASS；新活动状态只记录实际当前源证据。不要移动当前manifest/当前task-104产出到旧报告归档，先判它是否本轮事实。
5. CLEAN-CODE先找真实import/注册/调用引用，再删除旧实现，更新__init__/PassManager与测试集合；不允许正常入口import archive或try/except回退旧链。不为通过import而返回空Module/identity evaluator/伪VERIFIED。
6. 清理验收：git diff --check；最小import/smoke；正常入口可达且无旧分派；随后MAIN-A/B完成符号/数值/拒绝矩阵后再运行本轮门。仅import成功不是主链完成。
7. 名称含legacy/fallback/unsupported不单独构成删除理由：原v33 rows<=2数学分支、正确容量/同步/ABI拒绝、当前模板未支持的具名拒绝均保留；不得把正确性检查当兼容防御层删除。

已核事实限制：这是一轮文本/源码只读审计，不声称全仓无残留；未跑pytest/编译/设备。glob扫描workspace时报告无关run03缓存目录Permission denied，未访问或绕过它们；目标native_v3与主线所需文件可读。
