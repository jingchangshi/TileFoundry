# INPUT-FACTS（task-2，只读调查）— flex-1-g1

分析基线 `git rev-parse HEAD` = f341e47529453bdbc8dd81f2c108d36628fb4b8b；tag handoff-20261112 = 374de77（HEAD 多 314e016/f341e47 两提交，均只改 targets/ascend 下 10 文件；本文各结论 pre-f341e47 == post-f341e47）。全程只读、无 git 写、无设备调用、无 worktree。

## 1. work_package/ 输入完整性（vs work_package/PACKAGE_MANIFEST.json，82 条目，baseline f4d6fe1）
- 交接树：56/82 在盘且 sha256+bytes 全部一致，0 清单外文件；**缺失 26 = evidence/ 17 + implementation/ 6（4 个代码文件+README+__init__）+ plan/ 3**（清单行 38-121 / 222-251 / 252-266）。
- 原包 .../TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3：**82/82 sha256+bytes 全一致**、0 缺 0 多；无 .git（普通目录，非 worktree）。
- spec 明确要求迁移缺失三类：prompts/START_TEAM.md:18（按序读 implementation/README+4 文件+plan/ 三件）；START_TEAM.md「如何使用本包代码」:47-53（四文件迁入 vendor/pipeline/p3_chain）；specs/07_integration_interfaces.md §7.3 表 :30-38；MASTER_SPEC.md §2.5 :336,342、§2.6 :357、§2.7 :369（evidence/v3_schedule_audit.md）、§3.6 :389；specs/03:18；specs/05:440,445。
- 实测故障：work_package/tests/{test_compile_transaction.py:6, test_modulo_kernel.py:6, test_symbolic_template.py:7-9} `from implementation...` → 树内 ModuleNotFoundError；tools/verify_package.py（读 plan/TASKS.json，tools/next_tasks.py:8-9；plan/GATES.json :36-38；82 项清单 :42-50）在交接树 FileNotFoundError plan/TASKS.json，在原包 PASS{fixture 18, cases 40, tasks 34, gates 8, manifest checked}。

## 2. fixture / 用例身份（任务 B）
- work_package/fixtures 全部 20 文件与原包逐字节一致（diff 为空）；关键文件 sha256（与 fixtures/source_manifest.json 及原包同值）：mla/golden.py 3996ec1a…bc8cdc；mla/test_mla.py f1262275…8d981；mla/cases.yaml 57e742e5…df46d；mla/cases.csv 85648d43…5a8；mla/proto.yaml 70cd1163…10f4；attention_residual/golden.py cf90b847…795a；attention_residual/test_attention_residual.py c50fcf95…29559；attention_residual/cases.yaml d6d35f24…16792；attention_residual/cases.csv eddae5a9…b8a；fixtures/cannbench_contract.py 026f51a9…7400；cases/case_matrix.json d57785b1…4ef8（两树同值）。source_manifest.json 声明 copy_policy=byte_identical、18 文件、source_execution_status=NOT_RUN。
- case_matrix.json：AR 20 + MLA 20 行齐全，key 恰为 level4/{case}_1..20（=Harness Case.label），seed 公式 40/40 行成立；AR 输入 [v_stack,x,w,gamma]（首行 [4,1,128,2048]/[1,128,2048]/[2048]/[2048]，bf16）20 个不同 shape 组合、dtype 全 bf16 或全 fp16 两种；MLA 输入 [q_nope,q_rope,k_nope,k_rope,v]（首行 [2,128,128,512]/[2,128,128,64]/[2,128,1,512]/[2,128,1,64]/[2,128,1,512]，fp16）20 组合、bf16/fp16 两种；40 行 execution_status 全 NOT_RUN。supplemental_mla.json=4 条诊断用例，scope 明文不替代原 20。
- 阈值/比较器：包内 golden/test/cannbench_contract 与原包字节一致 + README_HANDOFF.md:44「Original contract fixtures and comparator thresholds are preserved; no threshold or comparator change.」；npu_compare 本体在包外，位置 UNKNOWN。

## 3. 原件可用性事实（不下裁定，D1 归 Lead）
- 原包可读：implementation/ 行数 modulo_kernel 362 / symbolic_template 117 / native_bindings 148 / compile_transaction 83（另 README 39、__init__ 5）；plan/TASKS.json 1095 行、GATES.json 188、PHASES.md 49；evidence/ 17 文件 2107 行。
- 禁令原文：docs/HANDOFF_PLAN.md Entry「Do not use the old TileFoundry repository/worktrees.」；README_HANDOFF.md Source「The original TileFoundry repository now retains only its main branch; do not use it as the new workspace.」、Inputs「historical evidence/implementation/plan are intentionally excluded」；docs/ACTIVE_CONTRACT.md 禁止依据仅指 _archive_compat_defensive/ 与 HISTORICAL_WORKTREE_DO_NOT_USE 标记的历史 worktree（本包目录无 .git、未被打标，深度 4 内未找到该标记）。
- 反向事实：artifacts/native_v3/research/mainline_cleanup_manifest.md 执行顺序 3「保留只读工作包 20261112/tf_ascend950pr_native_v3 及 reference 原样」。

## 4. src/ targets/ 残留引用（实际导入 333 个 .py，PYTHONPATH=src:targets/ascend）
- 330 OK；**会导致 ImportError 的既有条目仅 2 处且均不在 ascend 正常入口**：tests_native_mainline/backend_operators/test_operator_abi_numeric.py 顶层 `from native_reference import`（backend_operators/ 无 __init__.py，仅靠 pytest 路径注入成立；pytest --collect-only -q -p no:cacheprovider = 24 collected exit 0，与 Lead 24 passed 口径一致）；tilefoundry/orchestrator/causal_lm/run.py 缺第三方 transformers。另 tilefoundry/cli/__main__.py import 即跑 argparse（SystemExit 0，副作用非错误）。
- 无指向已删除 twin/materialize/unsupported-inventory 的 import（命中全是注释或 src/tilefoundry/runtime 的 RuntimeModule twin 概念；test_native_builders.py:167 断言无 attnres.twin）。parser/pipeline_intents.py 仍存在且被 parser_visitor.py:17 正常引用（旧清理清单曾计划删它，交接快照未执行）。
- 12 处 raise MainlineNotImplemented（plan/mainline.py×4、solver×1、storage×3、codegen×1、completion×3）tag 与 HEAD 相同；passes/p3_chain.py:201 无条件 NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED 缺口 tag/HEAD 均在。

## 5. 旧 worktree（仅 ls / git -C 只读）
- /workspace/TileFoundry_native_mainline 与 /workspace/TileFoundry_Ascend950PR_AgentTeam/TileFoundry_native_mainline：**均不存在**。
- /workspace/TileFoundry 存在：main @ f4d6fe1，worktree list 仅自身，origin/feat/tf-ascend950pr 未检出。/workspace/TileFoundry_Ascend950PR_AgentTeam 存在（20261112/ 含完整原包、artifacts/native_v3/）。
- 作为实现依据：旧仓库/worktree 被 HANDOFF_PLAN/README_HANDOFF 原文禁止；原包非仓库/worktree、未被禁令覆盖且被要求只读保留。

## 6. UNKNOWN
npu_compare 包外位置；20261112/ 目录内除 tf_ascend950pr_native_v3 外的其他交付物；source_manifest 引用的 attention_residual.tar/mla.tar 原档位置；evidence/ 内容正确性（只验 sha256 身份，未复跑 reproduce）。

复核：清单逐项 sha256sum；work_package 根跑 tools/verify_package.py；导入清点与 pytest collection 命令同上；全部只读。

## 7. Addendum（task-52/56 轮，lead 要求留档）— 两条自我更正与一条方法错误

- **更正 1（会咬 task-52 校验器）**：我先前给 implementer 的摘要写「校验器可用 ast.literal_eval 读 CANN_BENCH_THRESHOLDS」——**实测不成立**：literal_eval 对单个 `2 ** -10` 即抛 `ValueError: malformed node ... ast.BinOp`，对整表同样失败。正确做法 = **BinOp-aware 折叠**（left 常量 2、op=Pow、right 一元负号 → `2.0**exp`），非 pow2 分支取 `ast.Constant`；已实测对切片 [82,132] 可用。该坑已由 lead 转给 task-52，explorer-g1 已写进 DRAFT.md §3 并作为 JSON 存 source_forms+values 双份的理由。
- **更正 2**：`threshold_table.line_range` 应为 **[82, 132]**（82 是表开括号行，132 是闭合 `}` 行）；:133 属 `_FALLBACK_THRESHOLDS = CANN_BENCH_THRESHOLDS["float32"]`，不应并入表体。我先前提的 82-133 含 fallback 行。附带事实：`float8_e5m2` 末两档是 float 常量 `1.0`（非 `2**-n`），登记与校验均不得归一化。
- **方法错误（lead 纠正）**：07:41 我据 `find` 0 命中断言 COMPARATOR_REGISTRY_DRAFT.md 不存在并推出「task-56 不能判完成」；该文件实际 mtime=**07:41:10**，与我的观察落在同一分钟——结论建立在**落盘之前的瞬时快照**上。教训：**瞬时快照 ≠ 持续状态**；**第二次猜测仍是猜测**（把同一条命令再跑一遍而不重新核对 mtime/HEAD/时钟边界，只是重读同一个过期快照，不构成复核）。此后凡「缺失/不存在」类断言必须附观察时刻，下结论前复查 mtime 与 HEAD 是否已前进。本轮已按此执行：addendum 前重查得 HEAD=b6d8848（registry 已入 883117e、DRAFT.md 已入 b6d8848），两文件 sha256 与字节数（b25dd485…/44239B；073bc999…/5141B/58 行）与我对 a/b/c 复核时一致，explorer-g1 报告的数字与我的独立复算吻合。
- a/b/c 复核结论不受上述影响，且复核未把 registry 的 values 当真值（独立重折叠后逐位比对）：44/44 条 source_forms+values 与磁盘表一致、dtype 分布 AR bf16×17+fp16×3 / MLA fp16×15+bf16×5 / suppl 2+2 与源对撞无一例外、全文 PASS/通过/VERIFIED/certified 命中 0（唯一 passed 在否定句）。
- 归属纪律：task-56 两文件均为 owner explorer-g1 落地与收尾（板上 completed rev3）；我只读复核、不判他完成、未改其文件。task-52 因 required_role=implementer 我从未认领（TEAM_ROLE_MISMATCH 即停），交付的是事实包 + 上述更正。
