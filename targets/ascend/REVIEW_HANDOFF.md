# Ascend950PR 外部审阅交接：INCOMPLETE，方案实现暂停

**这次合入是把现状、未完成实现、诊断和反例交给外部模型审阅，不是项目验收通过。** 用户已暂停方案实施；当前授权仅用于整理提交和提供审阅材料。不要根据旧证据 JSON 的 `VERIFIED`、测试数量或静态微例正确推断完整方案已实现。后续方案由用户另行提供，本交接不擅自替换它。

## 给远端审阅者的最短入口

1. 先读本文的合同和已知缺口，再看 [完整 HIR/TIR/TL 审阅材料](evidence/native_aiv_review_20261112/REVIEW.md)。该材料属于固定四个64元素f32 tile 的两输入乘法微例，不是完整 AR。
2. 运行环境的新事实见 [CANN92汇总](evidence/cann92_runtime_20261112/summary.json)、[确切命令](evidence/cann92_runtime_20261112/COMMANDS.md)、[禁缓存运行复验](evidence/cann92_runtime_recheck_20261112/split_result.json)。
3. 历史过程和反例见 [checkpoint](evidence/G61_G70_RECOVERY_CHECKPOINT.md)。它是时间顺序记录，后文会更正前文；本文与新运行原始日志优先说明当前状态。
4. 本分支保留此前238个历史提交，新增收口提交按范围分组；不 squash/rebase、不重写既有历史。`main` 合并与远端推送由 Lead 单独负责。

## 本轮必须遵守的核心合同（无需本机工作包也能审阅）

- 两个目标算子是 `attention_residual` 和 `mla`；每次调用恰好一个逻辑计算 kernel。MLA 的配对 AIC/AIV 程序可以属于同一逻辑调用，不能机械按 profiler 硬件行数判断。host 仅作元数据校验、输出分配、specialization选择和一次调用；不得用额外 Torch/预处理 kernel 完成 concat、transpose、cast、matmul、softmax 或 residual。不得在 GM 物化完整 MLA scores/P 矩阵。
- 作者 Scheduled HIR 表达时间/空间/存储/指令策略；复用 native TF 类型、原 IR、分析缓存、PassManager、StoragePlan、指令注册和 runtime，不另建 JSON 业务驱动的平行编译器。进程内保存真实 Expr/IR 引用，JSON 仅作持久化与诊断。
- 调度 unit 是完整 VF、合法 DMA、Cube 或有可验证内部合同的模板。保留 For/If/Let、真实执行域、captures/carried/yields SSA；不能把整个底层指令清单无结构扁平求解。
- Z3 面向有限 task/region 问题求 start、II、未固定 versions；`start = stage*II + phase`。作者 stage、II、fixed versions、precision/core 和 issue_after 是硬约束。issue_after 只表示同物理步发射偏序，不等于数据依赖或完成；数据 distance/RAW/WAR/WAW/状态效果另行推导。max_buffer_versions 与 max_stage 是不同概念。
- **动态/符号 trip count N 不应按 N 复制整个 loop body。** 真正流水应保留符号循环，以物理步 p、逻辑 ordinal k=p-shift(task)、有效 guard、SSA 和 slot=k%V 表达 prologue/steady/drain；非零lower/step也必须保持语义。编译规模受有限 task/策略限制，不随真实大N线性展开。
- 唯一正式 `ResolvedPlan` 应消费 solver 的 start/II/versions，完成、存储和最终 TIR 使用同一来源。只有完整并验证的计划才能导出运行产物。
- READY 保证 producer写完成后读；FREE 等所有旧consumer读完成后才覆盖；配置状态与退出也需真实完成保证。generation是完整逻辑代次，不是取模slot。P、alpha、KV、output UB等必须活到真实最后consumer；短loop/N=0/尾部/guard一致性必须正确。
- TileLang backend必须执行 TF解析计划。仅允许语义保持的事件重编号、地址重定位；不能把同样tile/stage重新交TL自动调度后声称执行原计划。显式模式要兑现核心域、同步、版本和流水，真实编译 PassConfig 必须关闭重调度；源代码中的 `auto_schedule=False` 注释/赋值不构成证据。
- 正式数值合同是原20+20用例、原输入/seed/golden/helper/比较器，不能改阈值、删例或CPU替代。helper实际使用 cann-bench模式及precision_thresholds；最终dispatch必须完整重跑原40例，同时提供单逻辑launch、计划对应、资源与性能证据。
- G70搜索至少每算子8代表，每代表8个实质不同且数值正确的设备候选；每原case至少3个可行竞争候选，每代表winner有局部邻域。初始每case上限40（12 seed+20 neighbor+8 advanced），还有总时间/编译/运行预算。不同名称、随机seed、flag编号不算新候选；不能重复配置凑数。AR三族online/blocked_online/resident_n和MLA canonical/deferred/output-shard都不能因预算耗尽或未实现而豁免。
- deferred要求旧anchor下全部PV（所有列及hi/lo）完成→drain/prefix rescale/pending clear→新anchor PV。group predicate/factor按尺度代次保活并为组内统一决定，故障注入要拒绝旧尺度PV迟到和factor提前覆盖。目标能力或原比较器精度实验可支持具体路线不可行结论；缺设备、预算或缺代码不等于不可行。

## 原规格来源与远端可见性

原工作包不在本仓库，也不是远端链接：本机路径为 `/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/2026111/TF_Ascend950PR_TEAM_START.md`，规格目录为同目录 `tf_ascend950pr_single_kernel_v2/`。关键文件是 `specs/00_master_spec.md`、01/02/03/06/07/08、`plan/TASKS.json` 和 `plan/GATES.json`。远端reviewer无法假定访问这些本机文件；需用户提供完整包/新方案时再对照，本文已自包含核心约束。不要把本机相对路径当远端可打开文档。

## 当前真实完成与未完成边界

| 项目 | 当前判断 |
|---|---|
| native TF四块乘法微例 | CPU正式analyze/finalize/emit、显式TL、真实编译和分进程设备运行可复现；不是完整AR |
| 微例数值 | CANN92宿主product(x≠z)、square(z=x)均exact，max_abs_error0.0；禁缓存重编译/复验PASS |
| 完整AR/MLA生产入口与计划闭环 | INCOMPLETE，未闭合 |
| G61 | PARTIAL，只有CPU代数/epoch协议模型；没有真正deferred MixedKernel与native effects/completion接入 |
| G69 | INCOMPLETE；CPU生成器/过滤/模拟时钟诊断；独立worker deadline有CPU测试，未构成真实设备调优门 |
| G70/G71/G72 | 未完成真实预算搜索、dispatch、最终40例及复现交付验收 |

### 需要优先审查的实质缺口

1. **static展开偏离目标。** 当前微例使用 `plan/materialize.py` 的有限static展开：最终TIR在一次wrapper内含四份body。修复了每份真实GM索引与SMEM别名，但不能称为动态符号pipeline。实际生产P3链先走serial物化，非完整动态方案。
2. **形式上的symbolic结果未安装为实际入口。** 独立 `plan/pipeline.py` 的符号loop/SSA重写不等于production routing已执行；P3链的symbolic结果可能只记录在metadata而非替换module.entry_function。必须核真实入口、artifact及输出，不能仅对metadata断言。
3. **投影策略局限。** `passes/p3_chain.py` 当前投影固定stage0、buffer版本1、固定serial；dynamic/nonstatic域拒绝，尚未把作者四维策略和共享Z3完整解兑现到正常运行入口。不能把独立solver/变换单元测试当生产闭环。
4. **kind检查漏operand。** `codegen/native_aiv.py` 的操作种类/order/count检查仍会接受finalize后把Vmul operands `(va,vb,mask,dst)` 改为`(vb,vb,mask,dst)`，发射z*z而不是x*z。内存access map不覆盖所有算术；完整每source_stmt+k的args/attrs/bindings fingerprint、多重性覆盖尚待实现。**严格算术对应未完成。**
5. **旧twin/checker弱。** 通用文本twin含T.no_op，若checker仅核注释/子串，可对真实slot/guard/wait错误返回MATCHED。AR文本checker也不是正式plan/TIR对应证明。原样保留供review，不视作验收。
6. **单进程ORTools crash。** 现native设备脚本在torch_npu加载后调用TF memory analyze，`ortools/sat/python/cp_model.py:1771 solve → allocation.py:1300 → memory.py:713/629 → analysis/api.py:205`段错误。初始化本身在9.2成功。分进程CPU emit + TL-only runtime通过，没有修复这个wrapper/库交互问题；不要混淆。
7. **MLA直接TL旁路。** `mla/kernel.py`直接tilelang.compile、固定T.Pipelined/versions，不能据旧数值日志宣称TF唯一计划被执行。输出shard grid由unsharded jobs限制；D448合法分片域不同；真实KV tail/fullmask/归一化及原runner证据仍有缺口，详checkpoint。
8. **后端受限域。** 新native桥仅支持两个输入/一个输出、256 f32连续GM、单AIV、4×64、full predicate、serial fixedV1，未知op/control拒绝。它不是泛用binder-aware/native AR后端。generic runtime bind/invoke/readback仍有显式UNIMPLEMENTED路径。

## 证据如何解读

G43/G50/G51/G52/G60/G62/G63等旧JSON中的VERIFIED是历史局部数值、模型或结构诊断，不等于本次目标gate的当前PASS。部分旧描述与后续审计矛盾，保留而非重写历史。G61此次明确PARTIAL；G69此次由旧顶层VERIFIED纠正为INCOMPLETE，历史模型状态另存。旧resident/busy理由是当时备注，不是当前CANN92环境阻断。

AR/MLA历史脚本可能来自仓库外 `mla-tl-run/`，手写20例+warm oracle cache不等于最终dispatch调用权威runner及产物身份绑定。原测试输入比较器、trace、conformance和性能各是独立证据层。

`native_aiv_review_20261112`保持当时编译PASS/deviceNOT_RUN的原审阅快照；后来设备成功仅记录在新CANN92目录。不要改旧快照伪装它当时已E2E。原CCE、raw日志保留，不含binary缓存、pycache/core。库maps只是本机实际加载路径，不是可移植环境；环境文本仅选取CANN路径。

## 已验证结果与复现（需要既有兼容环境）

本机使用已有 `../envs/tfnpu-v2/bin/python`（Python3.12.16，指向tilefoundry-npu-dev），Torch2.12.0+cpu、torch_npu2.12.0、TL0.1.15+cpu.git0e687c17；普通项目依赖也需安装。文中命令相对仓库根；远端机器需自行替换解释器路径，不要安装/替换驱动来迎合本文。

历史native组合：**52 passed**（Lead独立结果8.59s；当时测试数，后续新增反例数量会变化）：

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend ../envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests/test_native_aiv.py targets/ascend/tests/test_native_materialization.py targets/ascend/tests/test_p3_chain_g33.py targets/ascend/tests/test_scheduled_tir_g42.py
```

后续native支持域反例与现有TF HIR算术oracle：**13 passed in4.10s**，对应f61b2f6；仍漏上文Vmul重接线反例，不能解读为完整conformance。

worker deadline组合：**64 passed in4.62s，exit0**（Lead独立验证）。涉及真正子进程超时、父先退出而child忽略TERM、共享剩余deadline、过期/stale结果拒绝；只测CPU worker边界，不构成NPU候选搜索。准确命令：

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend ../envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider targets/ascend/tests/test_search_deadlines.py targets/ascend/tests/test_search_worker_deadlines.py targets/ascend/tests/test_search_g69.py
```

宿主运行使用CANN92：`/data/pri/cann-9.2.0/ascend-toolkit/set_env.sh`，realpath为`/data/pri/cann-9.2.0/cann-9.2.0-beta.2/set_env.sh`。AgentGates本机env.sh指定的`~/.conda/envs/tilefoundry-npu/bin/python`可init/roundtrip但无TL；有TL的dev解释器用于微例。旧adapter/script的container-only注释与host92新证据并不一致；未为本收口修改它们。

同源静态微例（已有设备授权时才运行，全程设备锁；不是完整方案验收）：

```sh
source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TILELANG_DISABLE_CACHE=1
flock -w 5 /home/.tf_device0.lock timeout --signal=TERM --kill-after=3s 120s ../envs/tfnpu-v2/bin/python -X faulthandler targets/ascend/evidence/cann92_runtime_recheck_20261112/runtime_only.py
```

原raw复验exit0，明确TL begins/completes freshcompile、product/square各0误差。source SHA256=`022e44a121c84bcc864f99b7483f56508471506662420bf488f8210109f1a5e9`；CCE SHA256=`97169f6eef0a98bd426c8d6455406dc2cc065a00874ede2149f15f37b1ab8fe5`；formal CPU TIR digest=`936c5b3ca6eb44f1a5d3466220446a429024a00c6acf7c98cd0f0de358913da3`。runtime进程不导入TF/ORTools，CPU emit进程不导入torch_npu，说明分进程隔离有效，未说明单进程已修。

## 后续审阅与动作边界

请优先评价正常入口是否真正消费符号IR/solver解，完整计划是否安装为实际执行体，以及上述反例能否被真实验证器拒绝。不要围绕已暂停的静态展开路径继续包装性能完成状态。待用户提供/确认新方案后再实施；本次main发布仅供外部审阅。

仓库安全约束仍有效：不reset/clean覆盖用户改动，不sudo、不重置设备、不杀他人进程。NPU唯一执行者持`/home/.tf_device0.lock`；遇疑似挂起停止本任务，不能自行reset。没有持续运行的后台设备job。
