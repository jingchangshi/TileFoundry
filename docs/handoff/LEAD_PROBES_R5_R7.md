# Lead 独立探针记录（R5/R6/R7 阶段）

基线 HEAD f341e47；解释器 /home/s00653124/workspace/envs/tfnpu-v2/bin/python；三环境前缀。以下全部由 Lead 亲跑（脚本 /tmp/probe_*.py），用于替代反复中断的 reviewer/validator 长任务，不替代其后续独立复验。

## R5 delay 登记表
- problem_identity baseline sha256:462c217…；_OCCUPANCY[gm_to_ub].ticks+1 → c7a9e59…；status→DEVICE_VALIDATED → 231c58c…；恢复后回到 baseline。delay_table_bytes 同步变化。登记表是身份承载的：成立。
- pset 不在 _OCCUPANCY，在 _MODELLED_GAPS：MODEL_COST/ticks=1/source 明写 ISA 未声明。units.py 的 max(1,·) 伪造路径已消失。
- vld/vsts 与 PIPE_FACTS 的冲突以 fact.uncertainty='UNRESOLVED:…' 记录，不静默解决：成立。
- 外层边实测：independent_ub 64/64/4（DECLARED/OCCUPANCY_FALLBACK/MODEL_TICK）；minimal_aiv 64/64。两例 reuse 均为空（无第二次覆盖）。

## R6 proven-zero 谓词
- Lead 发现的盲点（同 core_kind 不同 sid 被判零、core 默认断言 aiv）已实现为必填参数。复测全部符合：
  same instance zero: True / diff sid zero: False / diff sid fallback: DECLARED-OCCUPANCY_FALLBACK-3（vadd）
  grid 2/0 → False；grid None/'x' → DelayRefusal DELAY_GRID_UNSOURCED；owner None → DELAY_OWNER_MISSING；vector→scalar → False。
- frame 检查精度：正常 independent_ub require_kernel_frame 放行（1 frame、4 allocs），不是无条件拒绝。

## R7 死路证据
- resolved.py:343 在 source audit 后**无条件** raise：independent_ub 实测 region:0 SAT/II208 之后被 PLAN_DEPENDENCE_PROOF_UNAVAILABLE 拒。ResolvedPlan 永不存在 → completion/storage/apply/TL 正向全不可达。architect 已把'无条件拒绝≠fail-closed'写入 ARCHITECTURE_AND_PLAN.md（217 行，sha256:ba263253…），S0 第一项里程碑改为'independent_ub 产出 ResolvedPlan'。修复由 task-44 承担。
## R8 阈值来源实测（Lead 亲跑，推翻 task-42 的'\''阈值不可得'\''结论）

比较器在 handoff 指定路径**存在且自带 per-dtype 阈值表**：
- /home/s00653124/workspace/BiShengKernelAgent/skills/common/ascend-npu-run-eval/scripts/npu_compare.py:7-8 直接 import compare_dtype_close_case / compare_cann_bench_case，按 accuracy_mode 分派（:37-41 resolve_accuracy_mode）。
- 合同路径用 accuracy_mode='\''cann-bench'\''（cannbench_contract.py:982-1006，explorer 已实测），分派到 cann_bench_compare.py 的 CANN_BENCH_THRESHOLDS（:82-133）。

阈值表（原文）：float16 threshold=2^-10 / small 2^-11 / small_err 2^-16 / cancel_boundary 2^-5 / cancel_zero 2^-5；bfloat16 2^-7 / 2^-8 / 2^-16 / 2^-3 / 2^-3；float32 与 float64 2^-13 / 2^-14 / 2^-30 / 2^-8；**hifloat32 2^-11** / 2^-12 / 2^-28 / 2^-8；float8_e4m3 2^-3；float8_e5m2 2^-2。_FALLBACK = float32。
判定语义（cann_bench_compare.py:10-45）：shape 必须相同；bool/整数默认**精确**（abs diff <= max(extra['\''precision_thresholds'\''][dtype],0)，缺省 0）；NaN 位置必须一致；共享 Inf 同号；单侧 Inf 用 signed finfo_max 替代；relative_error=|a-g|/(|g|+1e-7)；**通过 = MERE < threshold 且 MARE < 10×threshold**；small/cancel 各自独立 pass flag（small/cancel 允许 ≤2× native error）。
关键：`thresholds = table[actual.dtype or float32_fallback]` 且 `target_golden = cast(golden, actual.dtype)` —— **阈值由 actual 的 dtype 选择**。因此 f32 kernel 对 bf16 golden 会按 float32 容差判，而不是原例容差：这从机制上证明了'\''f32 代验不能冒充原 40 例数值通过'\''，不再是推断。

结论对门禁的影响：G3 的'\''阈值来源不可得'\''不成立，改为**登记即可满足**：来源=该比较器模块路径 + CANN_BENCH_THRESHOLDS 表 + accuracy_mode='\''cann-bench'\'' + 版本/身份。禁止任何修改比较器、改阈值、或往 proto.yaml 塞 precision_thresholds 的动作（原 40 例与比较器/阈值不变仍是硬约束）。proto.yaml 缺 precision_thresholds 只影响整数/bool 的'\''精确'\''语义，不影响浮点判定。
另：表里有 hifloat32 键，提示 hi/lo 语义在比较器侧有登记 dtype——MLA p_hi_lo 的容差来源应查这一键，而不是自造。

## R9 条件门强度（Lead 亲跑，/tmp/probe_r9.py、/tmp/probe_basis2.py）
- 证明确实**来自实际 IR 重读**：source_audit.py:235-241 从 problem.source_snapshot.entry_ref 走 read_static_ub_windows + 真实 accesses/effects，而不是从 problem.edges 反推。
- 协调篡改被拒：反转真实边 + 把 coverage.problem_identity 改成全 0 字符串 → **SOURCE_DEPENDENCE_MISMATCH**（edge direction or source…）。自报 hash 不能放行。
- 多 region 不空真：dependence_proof 在 len(regions)!=1 时返回 missing 明确点名（:479-495），不靠"没有缺陷"充当完整。
- basis 参与身份：occupancy 行 basis DECLARED→MODEL_COST（ticks 不变 64）→ plan_identity a03843c… → 2e73087…；恢复后回 baseline。
- 未声明永不投影数值：occupancy 改 UNSOURCED/ticks=None → **DELAY_UNSOURCED** 拒绝（不是变 0）。
- 更正我自己的一个探针错误：第一次用 dataclasses.replace(basis=D.DECLARED) 对 gm_to_ub 是**空操作**（该行 basis 本就是 DECLARED），因此误得"basis 不参与身份"。第二次改成 MODEL_COST 才真正变化。教训已用于后续：改字段前先打印原值。

## R10 稳定性与范围（Lead）
- 非 flaky：test_tl_emit_read 单跑连做 4 次均 9 passed；全量 152 passed。之前 4→8 的振荡是两位实现者在我两次探针之间分别落盘所致（函数改动与调用点/断言未同批落盘的典型表现）。纪律已重申。
- fp16 定夺（architect，写回主文档 220 行内）：fp16 是本目标**应支持**的输入/输出 dtype，不得记 TARGET_UNSUPPORTED；G1 的 fp16 缺项为三项（f16 载入与 transport、f16↔f32 双向转换、f16 输出存储），计算与状态走 FP32，native f16 直接计算在无设备证据前不声称。S4 范围，不阻塞 S0。
- 真实 bf16 作者程序：可降低（FINALIZE OK，params bf16 GMEM 两端，我独立复跑），但进不了发射支持域（TL_BUFFER_SCOPE_MISMATCH: carrier scope 未被承认 + vcvt/pset16 无 TL_SURFACE）。16-bit lane x128 是部署工具链 2048//bits 的寄存器几何、自标 UNSOURCED，待硬件证据。

## R12 身份分层与失效边界（Lead 亲跑消融；**更正 Lead 先前归因**）

先前我写"ISA/模型变更 → target_bytes → source → problem → plan"，**机制判断错误**，实现者的消融结论才成立。我独立复跑 /tmp 消融（删/恢复 ISA 五条 event 行）确认：source 60851B、target 18458B、model_version、problem、plan **逐字节不变** → 事件契约不进入任何身份层（target 实例不发布 ir/isa.py；字节里的 FlagSet/sync/completion_ops 来自 target.completion_ops()，step1 前已存在）。

我先前看到的 +51B（60862→60913）与 plan_identity 漂移的真实原因是两个**已修的真实缺陷**：

1. identity._fact 把 UnsupportedCapabilityError 的人类句子编进了 target/source 字节——换措辞即改身份（措辞不是程序内容）。现改为结构化 _tag("declined", fact class name)（identity.py:419/:444-445）。我实测 target_bytes 中 'Facts projection'/'UnsupportedCapability'/'no Facts' 全为 False，'TopologyLevelFacts'/'declined' 为 True（后者是结构化记录本体，合规）。
2. SOURCE_SNAPSHOT_DESIGN §4 要求 model_version 进 problem 内容，而 canonical 缺该键 → 改 ISA latency 时 model 变、problem_identity 不变。已补 "model" 键。

正确的**分层失效边界**（我实测复现，作为 runbook）：
- target facts（实例参数/容量/池/拓扑）变化 → source→problem→plan 既有身份全失效，须重新 analyze+resolve。
- 模型值（ISA latency_model、delays 表 ticks/basis/status、edge delay_fact）只经 model_version 与 delay 摘要进 problem/plan 身份，**不改 source/target 字节**。实测：ISA vadd latency→999 ⇒ src/tgt same、model/problem/plan changed；delays ticks+1 ⇒ problem/plan changed；恢复即回 baseline。
- 两者都是保守失效，不是语义变化；**不是 bug**，别再按 bug 追。
- 陷阱：delays._OCCUPANCY 在 import 时快照自 ISA，运行时改 ISA 不改表——模型变更必须重新导入+重新分析。
- independent_ub 当前真实值：II=208、starts 0/64/68/132、V=1、source 60851B、target 18458B、problem sha256:026fa446954c、plan sha256:1572729141f1、proof 完整（3 边/9 读/8 写/9 buffer/reuse 派生空）。

## R11 事件面与编译产物（Lead 复核）
- 事件可发性两条实测事实（已进合同）：TF 树只有 FlagSet/FlagWait/Drain（src/tilefoundry/ir/tir/flag.py），pipe_barrier/threadfence **无 TF op**（只有 TL 语言面函数）；TL 面**无 ascend_drain**。故 S0 支持域只接 flag_set/flag_wait，drain 记具名缺口，不得用无 TF op 的指令替代。
- 事件契约的 latency_model/ii_model 一律 None，delays 表对五条全给 UNSOURCED，resolve() 必 DELAY_UNSOURCED——不造 0 或 1。
- 真实编译产物（后端实测、Lead 认可判据）：kernel.aibin（ELF64 / machine 0x1029、arg_size=16=2×8、.Kernel_Type/.AIV_Kernel_Type 符号、**无 .CrossCoreSync**）+ executable.so + device_kernel.asc + params.json + manifest（逐文件 sha 核对）。这闭合了 task-45 的旧矛盾：最小 f32 kernel 源码里 flag 计数为 0，是因为我们根本没有事件，不是后端代补同步。
- 产物归属按内容配对（Ascend C 的 sha 必须等于本次返回文本；aibin 用部署的 make_key 重算归属，绝不按 mtime 猜）；KERNEL_PRODUCTS 与 BINARY_PRODUCTS 拆分（aibin 不在 kernel 目录，旧写法必然误报 INCOMPLETE）。
- 控制节点边界（Lead 加锁）：被动打开 /dev/davinci_manager、/dev/hisi_hdc（由 import tilelang 造成）→ 只记录并留取证方法；我们程序主动打开 /dev/davinci<digits> 计算节点 → 拒绝；validator 自发的设备动作仍在 flock 下。此区分须写成测试，不得只留注释。
- 环境前提更正：撤销"TileLang 与 OR-Tools 同进程必段错误"。实际触发为 sourced 环境 + 先 import tilelang（被动拉 torch_npu）+ 后导入 ortools sat → rc=139；反向顺序 rc=0；含完整 compile 的两种顺序均 rc=0。约束降级为导入顺序注意事项；文档中残留的"必须子进程"表述必须删除——错误前提留在文档里比没有文档更贵。

## R13 16-bit 认证洞（Lead 实测确认，非转述）
- 后端实现者主动上报：bf16/f16 进入 ELEMENT_DTYPES 后 16-bit kernel **可发射**，而 reader 尚无对应 gap。
- 我实测确认：codegen/tl_reader.py 内 grep 'provenance'/'UNSOURCED'/'SOURCED' **零命中**；三态判定只在 :690-696 依 refusals/gaps。=> lane 模型自标 UNSOURCED 时，16-bit 仍可拿到结构 VERIFIED。
- 裁定：补 gap（TL_LANE_MODEL_UNSOURCED，判定读 VectorModel.provenance **结构字段**而非其说明文字），不收回 f32——收回只是隐藏洞，且会删掉已实测可用的能力。原则：**能力可以有，认证必须有据**。
- 附带口径：32-bit SOURCED 的依据是交叉核对（2048//32=64 == target 声明 lane_count），不是默认信任。发射支持域扩大必须在报告中标"可发射但不可认证"，G2 未闭合。

## R14 身份内容覆盖缺口（Lead 实测，task-47 的依据）
- 方法：逐字段 dataclasses.replace 篡改真实 independent_ub 问题，同时量 (1) problem_identity_of 的 digest 是否移动 (2) resolve 结果是建成计划还是具名拒绝。/tmp/probe_fields.py、/tmp/probe_sweep.py。
- 事实：problem_canonical_bytes 的 regions 段只有 id 列表（problem.py:1098-1106）。**14 个字段既不绑身份、也没被审计层拦住**：task.latency、buffer.slot_bytes、buffer.generation_delta、access.generation、access.operand_index、effect.state_ref、effect.guard、region.max_stage、region.enable_offset、region.ii_min、region.owners、region.pipes、region.capacities、region.manual_pipe_order。
- 对照组（不绑身份但被 IR 重读抓住）：access.mode → SOURCE_ACCESS_MISMATCH；effect.pipe/effect.instruction → SOURCE_EFFECT_MISMATCH；edge.delay=-1 → DELAY_INVALID；edge.distance=1 → DEPENDENCE_DISTANCE；coverage.refusals → PLAN_COVERAGE_INCOMPLETE；target_identity/model_version 手填 → PROBLEM_CONTENT_CHANGED。
- 后果：completion 消费 access.generation/buffer 域/slot_bytes，storage 消费 capacities/slot_bytes —— 这些字段不同的两个问题会拿到**同一 plan identity**，身份不再描述它认证的语义。这是"记录自证"的根形状：proof 的自证已修（task-44 #21），但 recipe 覆盖面从未被逐字段验证。
- 方法教训（两条来自后端的自曝，同样适用于我的探针）：反向标签测试第一版写成 `_replace(model, provenance=...) if False else model` —— 空表达式，什么都没测；散文 prune 写在 digest **之后** —— 等于没剥，措辞仍驱动身份。凡"篡改后应被拒"的测试，必须先证明篡改确实改变了被绑定内容。
- 裁定：task-47 优先于 task-46；只扩展**同一个 recipe**（禁止第二处哈希），递增 IDENTITY_ENCODING_VERSION 作失效机制（非版本分派），并以 fail-safe 覆盖测试枚举每个数据字段（默认参与绑定，白名单仅限显式诊断字段 + 成员检查）。
- 事件面附带裁定：flag.py 的 slot/generation/versions 是 int attribute 且校验 slot==generation%versions；S0 直线程序实测 9 个 site 的 generation 全是 Constant，故 S0 用具体 int 即可、不动 core IR；符号代走控制 envelope（spec §4.3 的 if in_domain(k)），留待 task-46 之后单独立项。(c) 新增 Expr-only 事件 op 已否决（第二套事件模型）。

## R15 循环 region 的 carried 边缺失（Soundness，非"未证明"）
- 实测 minimal_aiv：region:1 只有 (unit:1→unit:2, RAW, d=0) 一条边；而 unit:2 上同 root 的读写对有 **base/left/right/mask/acc 五对**。base 的绑定在 For 之外（跨迭代存活），k+1 的 Vld(base) 必须由 k 的 Vsts(base) 满足 ⇒ 应有 carried self-edge unit:2→unit:2, d=1。
- 根因是**两处复制的同一假设**：problem.py:597 `left.task_id == right.task_id: continue`（edges 丢弃）与 problem.py:673 `access.task_id != last_reader.task_id`（reuse 的 next_writer 候选里排除同 task 写者）。于是跨迭代 root 的读写配对与版本回绕义务**双双不可见** —— 不是具名拒绝，而是根本没有义务，属静默丢失。
- 后果：core 已完整支持 distance 语义（core.py:313，且 210-214 为 carried 边算 latency//distance 下界），缺边时 solver 拿不到 t_dst + 1·II ≥ t_src + latency，**II 可被解到小于循环体真实需求** ⇒ 产出的是 unsound 计划。
- 当前被两道门**潜伏**：多 region 走 PLAN_DEPENDENCE_PROOF_UNAVAILABLE，单 region 的 For 走 SOURCE_CONTROL_UNSUPPORTED。因此放开门与 carried/d 推导**必须同批落盘**，否则会产出"看似已证明、实际约束不足"的计划——比现在的拒绝更糟（已下达为硬性要求）。
- delay 裁定：distance!=0 已排除在 proven-zero 之外，carried self-edge 必然落到 role→occupancy→model cost 阶梯。但 occupancy(vsts=4) 只是**写可见下界**，而 carried slot 复用需要的是**上一迭代末读者的读完成**，且 vld 的 read completion 正是我们记 UNRESOLVED 冲突的项 ⇒ 大概率 UNSOURCED → projection 具名拒绝。**这是正确结果**，禁止在 problem 层给它编数值。
- 要求测试形状：(1) 结构完备性与 delay 可证性无关 —— 每个跨迭代存活 root 的 RW 配对必须"成边或具名拒绝"，不得无声；(2) 差分证明 —— 删 carried 边 ⇒ SAT 且 II 低于循环体真实需求，补边+可信 delay ⇒ 同 II UNSAT；(3) d 只从 For step==1、root 绑定点相对 For 位置、循环内读写次序三者推出，缺一即 PROBLEM_DEPENDENCE_DISTANCE_UNSUPPORTED，禁止填 0 或嵌套深度；判据须**逐 region** 成立。
- 排序：task-47（身份内容绑定）先于 task-46 落盘 —— 46 新增的 carried 边与 d 值本身必须被绑进身份，否则会再次得到"身份相同、语义不同"的两个计划。

## R16 静默丢失审计 + Lead 两条错误结论的撤回
challenger 第三次中断，审计由 Lead 读码完成。确认的 D 类候选（已派归属）：
- D1 coverage 对 attribute-only op 的 effect 不计（problem.py:716-717 只在 input-kind 参数带 MemoryEffect 时加 expected_effects；FlagSet/FlagWait 全 attribute、Drain 无参数）⇒ 事件语句作为 task 被计数，其 effect 义务不被计数。咬合点在 step2 之后。
- D2 problem.py:577-578 task_order.get(access.task_id, 0) 默认 0 —— 第三次出现同形状（前两次是 reuse 端点与 edge 端点回填 0）。要求：不可达则改断言，可达则具名拒绝 + 回归，不接受保留默认 0。
- D3 problem.py:669-670 not readers or not writers: continue —— '没看见'与'确实没有'同一分支；要求用 universe 双向检查分开，并把'推导为空'的结论绑进身份。
- C1 completion EXIT 的核对机制：删除 EXIT 通道会被 COMPLETION_SITE_UNBALANCED（:383-384 site 必须作 producer 或 consumer）或 COMPLETION_EXIT_SITES_MISMATCH（:400-404 exit_site_ids 必须等于 REGION_EXIT 集合）抓住；机制是 site 平衡，不是 READY/FREE 式义务双向核对。要求补一条显式回归钉住机制。

Lead 自身两条错误结论（已当场撤回，保留原文以供方法复习）：
1) '悬空 channel producer 静默逃逸'（verify.py:241-243 的 continue）—— 错。同名代码在 :366-369 的 _verify_structure 里已 COMPLETION_CHANNEL_ENDPOINT_UNKNOWN 先拒，调用顺序 :308 结构 → :309 对应，故 :241 不可达。失误原因：看了 241 就下结论，未 grep 同名变量在文件里的其它出现。
2) 'EXIT 只由结构检查草草覆盖' —— 收窄为'机制不同但同样被拒'。失误原因：把'不在 READY/FREE 双向核对集合里'当成'没被核对'。
方法补条：报'静默丢失'前必须先确认该分支在**调用顺序上可达**，并 grep 同一标识符的全部出现；不可达分支的 continue 不构成soundness风险。

## R17 EXIT 义务可被一致删除（**更正 R16 中 Lead 的 C1 结论**）
- R16 我写"删除一条 EXIT 通道会被 SITE_UNBALANCED 或 EXIT_SITES_MISMATCH 抓住"。后端 owner 的四种形态实测推翻了其中的 (b)：
  (a) 只删通道、保留两 site → COMPLETION_SITE_UNBALANCED
  (c) 删通道+两 site、exit_site_ids 留旧 → COMPLETION_EXIT_SITES_MISMATCH
  (d) producer 指向不存在 site id → COMPLETION_CHANNEL_ENDPOINT_UNKNOWN（结构层 :366-369 先于 correspondence）
  (b) 通道+两 site+exit_site_ids 三者**一致**删除 → **ACCEPTED，无任何一层抓住**
- 根因：SITE_UNBALANCED 与 EXIT_SITES_MISMATCH 都是**内部一致性**检查，不检查**义务是否应当存在**。三者一起删就不矛盾。
- 影响面：S0 的 GMEM 边界异步写（STORE 后区域内无读者）依赖 EXIT drain 兑现完成义务；该义务可被一致删除 ⇒ 事件层认证强度低于 READY/FREE。
- 闭合：task-48 —— bind.py 独立推导 EXIT 义务 (EXIT, region_id, task_id)（WRITE_VISIBLE 的 GMEM 边界写且区域内无后续读者），与 builder 的 _drains 作为两条独立推导必须一致；_correspondence 纳入双向核对；(b) 分支由 assert ACCEPTED 改为 assert 具名拒绝。
- 同形状隐患（R16 D 类）：hard-event 池查询返回 None 不得退化为"无上限"，须具名拒绝；core 拼写只能取 core_for_pipe 返回值，禁止 upper()/lower() 手拼（实测 AIV="aiv" 小写，曾返回 None）。
- 方法条目（第三次同类）：**"某个缺陷被抓住"的结论必须按删除/篡改形态逐一实测**，不能由一条通过推广到全部；一致性检查 ≠ 义务存在性检查。

## R18 initial_values 绑法：Lead 的 (b) 方案被实测否决（第三次撤回）
- Lead 提议 (b)：把 carried initial 移到来源语句 occurrence，由 source audit 重读值。
- 实现者的反证（/tmp/p47_env2.py、/tmp/p47_pool.py）+ Lead 独立复核：authored 对象是 tilefoundry.ir.hir.function.Function（body 为 MeshRegion），不是 PrimFunction；identity.source_labels(authored) 实测抛 "only a real PrimFunction carries TIR label"，而 lowered 函数有 53 条 label。⇒ S0 现状下**没有可指的 TIR occurrence**，(b) 要成立必须为 authored Function 另造第三套定位机制，正是我自己反对的"各造一套"。撤回 (b)。
- 采用 (a)：每 region 用 summary 自带池（carried_params/captures/yield_values/induction_var）按**顺序**构造 binder 表（label 取位置不取名字），表与其摘要一起进 problem bytes，值在表内编码，表外 Var 一律 IdentityRefusal 点名路径。task-46 的 carried d 推导复用同一张表。
- Lead 追加要求（防止今天的"恰好为空"变成明天的静默丢失）：uncovered 非空必须具名拒绝，且须有回归 —— (i) 同位置同型改名 bytes 不变；(ii) 表外 Var 具名拒绝且 path 可见；(iii) binder 表少一条 → problem_identity 必变。

## 本阶段方法条目汇总（由三次以上同类失误归纳）
1. 跨层机制问题只信消融数字，不信任何一方的解释（包括 Lead 自己的）。
2. 报"静默丢失"前先确认该分支在调用顺序上可达，并 grep 同一标识符的全部出现。
3. "某缺陷被抓住"须按篡改/删除形态逐一实测；一致性检查 ≠ 义务存在性检查。
4. 篡改类测试必须先证明篡改确实改变了被绑定内容（空表达式、空操作 replace、identity 未变都算假清白）。
5. 断言前先确认被查询的东西真的非空（大小写/拼写导致的 None 不得退化为无约束）。
6. 函数改动与调用点/断言同一次落盘；跨分钟红会冻住全队与 collection。

## 方法条目 7（Lead 自我更正）：区分"持续状态"与"落盘竞态"
- 05:49 我实测到 plan/problem.py 的 IndentationError 与 test_tl_emit_read.py 的 unterminated string → 整个目录 collection error（8 errors），随即向两位 owner 发全队告警并夸大为"全量一条都跑不了"。
- 05:50 复核：整树 compileall exit 0、可收集，真实状态是 34 failed / 168 passed（task-47 在途的语义红，不是解析事故）。两处"语法错误"都是各自单次落盘被我的探针切成中间快照。
- 采纳的团队基线：宣布完成前先跑 `python -m compileall -q targets/ascend src/tilefoundry` + `python -c 'import tilefoundry_ascend'`（3 秒），并把"一次落盘"限制在单条 write/edit 内。
- Lead 自身约束：量到 collection error 或跨分钟剧变的结果，**先隔 20–60 秒复测一次**再发全队告警；告警措辞按证据强度分级（瞬时快照 < 稳定复现 < 可复跑最小例）。

## R19 task-48 复验（Lead 亲跑，含一次我自己的假阴性）
- 全量 **211 passed**（34 条同根红随 task-47 落盘消失）。exit_obligations 确实独立推导：bind.py 内 153 定义、297 消费；plan.py 的 _drains 不引用它（grep 零命中），符合"两条独立推导"要求。
- (b) 形态我独立复现：删通道+两 site+exit_site_ids 并重算 identity → **COMPLETION_OBLIGATION_MISSING**，locator 明确给出 'EXIT; region:0; unit:2'。原 ACCEPTED 已闭合。
- **我自己的第一次探针是假阴性**：我删了形态没重算 identity，于是被 COMPLETION_IDENTITY_MISMATCH 先挡下，看起来"已闭合"其实测的是 identity。这与我们刚立的第 4 条方法条目完全同类（篡改类测试必须先证明篡改改变了被绑定内容）。第二次按他们的方法重算 identity 才测到真正的义务层拒绝。教训：**他们的报告里写了方法（每种都重算 identity），我第一次没照做就下结论**——复验别人结论时，方法要一比一复现，不能只复现意图。
- 码名采纳他们的判断：复用既有 COMPLETION_OBLIGATION_MISSING，不新造 COMPLETION_EXIT_OBLIGATION_MISSING。理由成立：该码本义即"问题声明该义务而无边保护"，同义码只是多一个会被记错的名字。我先前建议的点名码撤回。

## R20 "被读就不欠 drain" 转移的可达性（Lead 亲跑三条篡改路线）
问：能否让一条 GMEM 边界异步写既不欠 EXIT、又没有 READY/FREE 保护？基线 independent_ub 有 2 条 EXIT 通道、write-only gmem buffer 为 buffer:7/buffer:8。
- 伪造一条新 READ access（同 root）→ **SOURCE_ACCESS_COVERAGE**（duplicate or foreign operand access）：access 必须由真实语句/operand 产生，凭空加的引用进不了覆盖账。
- 把真实 WRITE access 改成 READ → **SOURCE_ACCESS_MISMATCH**：audit 从实际 IR 重读 operand 效果，声明与指令不符即拒。
- 把生产 effect 的 completion 从 WRITE_VISIBLE 放宽成 issue → **SOURCE_EFFECT_MISMATCH**：effect 语义同样从实际指令读取核对。
结论：义务**不能靠篡改 problem 记录消失**；只有让那条写在真实 IR 里不再是异步边界写（即换了程序）才会不欠 drain，而那种情况由新程序的义务集自成一体判定。基线数字：211 passed。
剩余风险（未闭合，交给 apply 侧）：事件语句进入候选之后的跨层回归（删注入的 wait/set → verify_actual_program 具名失败）需要 task-46/step3 的 apply 注入，已由后端备好 reader 侧对手测试。

## R21 静态 0 trip 绕过与 stored 身份字段的权威性问题（Lead 亲跑，含一次自纠）
- 起因：把 R20 的 C 类（trip 静态读 0）打穿。方法：independent_ub 真 plan 上 replace rp0 的 lower/stop/step 使 region_trip_count 静态折成 0。
- 实测链：篡改后 build_symbolic_completion → **sites=0 / channels=0**（exit_obligations 首行 _static_int(region_trip_count)==0 ⇒ ()），义务全部静默消失；verify_completion 抛**裸 ValueError**（verify.py:190 解包 3 元组、expected 4 —— arity 不匹配，说明该路径从未被任何测试走到）；require_completion_verified → COMPLETION_NOT_VERIFIED（因为压根没验成功）。
- **Lead 的第一条结论是错的并已当场撤回**：我据 tampered.plan_identity == plan.plan_identity 断言"bounds 没被 plan recipe 绑定"，要求 implementer 在 task-47 补字段。随后直接调用 _plan_bytes(problem_ref, regions, proof) 重算：原始 vs 篡改 bounds 的字节**确实不同** ⇒ recipe 已绑定 lower/stop/step 值，不存在该缺口。
- 成立的结论是另一半：dataclasses.replace 会把 **stored** 的 plan_identity / plan_canonical_bytes 原样带走，而 completion 只读这两个字段、不重算 ⇒ 篡改者无需重算即可获得"身份没变、义务为零"。修复点是**消费者按同一 recipe 重算并比对**（PLAN_CONTENT_CHANGED 族），不是给 recipe 补字段。
- 形状归并：这与默认 0、空集合当没有、一致性当义务，是同一类错误的第 5 次出现 —— **stored 摘要字段不是权威，权威只能是对实际内容重新计算的结果**。任何消费 stored 身份字段而不重算的边界都按此处理。
- 新增方法条目 8：判"某字段没被绑定"之前必须直接调用做绑定的那个函数，不能拿被 replace 带走的 stored 字段相等来外推。
- 待修（已分派）：bounds 静态 0 必须在身份一致且实际 TIR 确证零执行时才可不欠 drain；裸 ValueError → arity 对齐 + 空 completion / 篡改 plan 各一条具名回归；所有 _static_int(...)==0 ⇒ 空义务的分支逐个自查输入可信性。

## R22 stored plan_identity 说谎不被拦（R21 的最后一块）
- 实测（independent_ub 真 plan）：replace(plan, plan_canonical_bytes=b"{}") → **COMPLETION_PLAN_STALE**（信息含 2 vs 10065 字节）。bytes 重算有效。
- 但 replace(plan, plan_identity="sha256:aaaa…") → build_symbolic_completion **接受**，把谎值原样搬进 completion.plan_identity；verify_completion 也 **ACCEPTED（sites=9）**，只有 completion.identity 与诚实版本不同。
- 严重度：不是立即授权漏洞（require 走内容注册表、不看该字段），但正是 R21 的形状 —— **stored 字段被读、未重算比对**。resolved.py:409 已有 plan_identity_of(plan)，重算一行即可。
- 要求：bind.py 补 digest 校验（stored plan_identity 必须 == plan_identity_of(plan)[1]），回归须覆盖三态：digest 谎+bytes 诚实 → 拒；digest+bytes 同时改成自洽的谎 → 拒（靠 bytes 重算）；两者都诚实 → 过。

## R23 fail-safe 扫描测试本身的两类假缺口（Lead 抽验）
- test_content_binding 两条红里点名的 12 处，我抽验四条：owners=() 、pipes=() 、resources=() 、Task.uses=() 逐个**丢弃** → digest 全部 MOVED；AccessRelation 换 o=i+1、may_access 取反 → 亦 MOVED。⇒ 那些"identity unchanged and nothing refused"多数是**扫描器 tamper 不 distinct**，不是真缺口。
- 关键洞察：替换型 tamper 常能过（换成另一个同型对象也 MOVED），**丢弃型（置空/丢字段）才是真洞的暴露面**。扫描器必须把"丢弃"作为正式一类，而不是只做替换；'unconstructible: no sibling for X' 正是只能替换的证据。
- 另一类：test_the_documented_holes_stay_honest 断言文档字符串里出现 'bound'/'caught'/'hashed' —— **测注释措辞**，应改为断言行为（该字段 tamper 被拒或被绑）。

## R24 architect 进度审计结论（Lead 采纳，方向已调）
审计发现（Lead 07:15 实测佐证）：gate 已从恒定码变成 10 条逐层具名缺项且**无重复行/无重复码**，但**全仓没有一条 ready=True 测试**（grep 'ready is True' / '"ready": True' 命中 0）。=> 十个条件门仍可能是**分布式死路**，只是把不可达藏得更讲究。R7 的方法论必须补一半：**不只是"无条件 raise 是死路"，"永远无法变绿的 gate"同样死路。**
architect 三项判定：① 主线未跑偏，但 **S0 收口标准被低估**；真的达成：作者入口→UnitTree→Problem→solver→ResolvedPlan、completion、storage（限无 rmem 正例）、emit/parse、离线 compile 与 binary ABI；未闭环：independent_ub 事件注入、apply/verify receipt、install、gate 同一真实对象链。② **防御性测试局部失衡**：23k 行中负例与身份扫描占比过高，32 红曾源于扫描器自身缺陷而非生产缺口。③ positive clearance 必须作为 **S0 独立必需验收**。
据此调整：
- 停止：继续扩大通用字段 fail-safe 扫描、同义拒绝码治理、注释措辞审计、把 rare 形状审计排在 positive 之前。
- 转做：WP1 S0 正向 gate（关键路径）、WP2 apply 事件注入、WP3 寄存器闸门、WP4 carried/distance、WP5 bf16 结构、WP6 比较器登记。
- 新硬判据（已下发给所有包）：**每个包完成必须包含至少一个真实正向对象链，或明确记 BLOCKED_ENV；负例只能伴随对应正例出现。**
- 两个最容易再跑偏的方向（architect 点名、Lead 认同）：把"增加拒绝/扫描测试"当进度；S0/S1 未绿就过早进 MLA/Cube/fp16 硬件面。

## R25 事件 op 的 mesh 契约冲突（Lead 裁定 (a)，改 core IR 三行）
- 现象（WP2 报告，Lead 实测复现）：verify_actual_program 在 TIR 合法性阶段拒事件 op —— T.flag_set 的 execution_mesh = thread_execution_mesh() ⇒ topologies=('thread',)，而 Ascend kernel frame 是 Mesh(("aiv",1), names=("core",))，matcher 不匹配，报 VERIFY_TIR_INVALID（判定点 src/tilefoundry/visitor_registry/instruction.py:539）。
- 实测解析链（src/tilefoundry/ir/pattern/utils.py:122-146）：常量 MeshPattern 原样用；topologies 为 WildcardPattern 时从 target 的 MemoryExecutionFacts.topologies specialize。GmToUb（Ascend，memory_execution_mesh）→ ('aiv',)；三种事件 op（core flag.py:96/112/128，thread_execution_mesh）→ ('thread',)。把 FlagSet.execution_mesh 换成 memory_execution_mesh() 后解析为 ('aiv',) —— 正是所需。
- 裁定 (a)：改 **src/tilefoundry/ir/tir/flag.py 的三行**（FlagSet/FlagWait/Drain → memory_execution_mesh）。依据不是"给 Ascend 开后门"，而是该模块 docstring 自述 TARGET-AGNOSTIC（"channel NAME, pipe names and flag id are opaque attributes supplied by the target"），而 thread_execution_mesh() 把拓扑名 'thread' 硬编码，是该文件唯一违反自身契约处；改成 wildcard 后任何 target 都按自己的 facts 解析，语义更正确而非更宽松。
- 驳回 (b)（在 kernel frame 内再包 thread MeshScope）：会给 Ascend kernel 引入它没有的执行语义，并污染 emit/ABI 的 mesh 结构（后续要靠它核对 loop/grid frame）——为绕开声明错误而改程序形状，正是审计点名的"用绕过代替修复"。
- 授权边界：只许改那三行 + 一条断言（三者 declared_execution_mesh(cls, Ascend950PRTarget()).topologies == ('aiv',)，且任一 target 得自己 facts 的层级）；flag.py 的 _check_common 校验逻辑不在本次裁定内。
- 附带确认：WP2 自查两处（sync 语句不计入调度单元计数；_Compare 先剥离 sync 再比对、收集后供集合核对）方向正确，必须留成回归——事件是"计划附加"而非源克隆。

## R26 本地检查点提交（用户批准）
- 提交 e7f0ca8 "ascend: S0 native mainline through ResolvedPlan, completion, storage, emit/read"（**本地，未 push**），父提交 f341e47。
- 内容：93 个文件、+27481/−560。范围严格限定 src/tilefoundry、targets/ascend、docs/handoff。
- **刻意排除**：docs/mainline_validation/HANDOFF_E_ORIG_IDENTITY_ACCEPTANCE.md 与 docs/mainline_validation/run-20261112T210101+0800-task136/ —— 属已停止的并行会话产物，不由本会话提交，保持工作树原样（git status 仍显示为 untracked，2 项）。
- 提交前实测：223 passed / 0 failed（提交后并行落盘已到 232 passed）。
- 提交信息里写明了已建立的真实性质（delay 三态、88 字段 fail-safe 绑定、EXIT 义务独立推导、receipt 不读自报字段、gate 分层、16-bit provenance 交叉核对、离线可产 kernel.aibin）与**仍闭锁项**（install/export fail-closed、minimal_aiv 需 carried 证明、rmem 事实未声明、原 40 例 BLOCKED_DEPENDENCY），便于后续会话直接接手而不高估进度。

## R27 G3 登记物独立复核（Lead 亲跑）
- docs/handoff/COMPARATOR_THRESHOLD_REGISTRY.json（44 cases，draft=true）：
  - 两个 sha256 与磁盘实际文件**逐字节相等**（threshold_table e347b921…、comparator ef2506bb…）。
  - 用独立 BinOp 折叠（Constant / USub / Pow / Div）重算 44 个 case 的 source_forms → values：**0 处不一致**；dtype 分布 bfloat16 24 + float16 20，恰等于 AR bf16×17+fp16×3 / MLA fp16×15+bf16×5 / supplemental bf16×2+fp16×2 之和。
  - line_range 正确记为 [82,132]，:133 单列为 cann_bench_fallback（此前 82-133 会把 fallback 行算进表体）；fp8_e5m2 末两档按常量 1.0 登记而非 2 的 0 次方。
  - claims_not_made / unknowns 明写：本登记物**不认证 G3 闭合**、不声称任何原例数值通过；并保留两条 UNKNOWN（S4/S5 实际导入的是否同一份 npu_compare 未证；比较器无 __version__，只能钉 commit+sha256）。
- 方法更正（flex-1-g1 实测，已转校验器作者）：**ast.literal_eval 读不了幂表达式**（连单个 2 的 -10 次方都抛 ValueError(malformed node / ast.BinOp)），必须 BinOp-aware 折叠。这是 registry 同时存 source_forms 与 values 两份的实际理由。
- G3 状态：登记物就绪但**未闭合** —— 仍缺 task-52 的只读校验器正例（草案校验通过）与设备侧运行时 __file__ 与 sha256 捕获。

## R28 关键路径收敛：gate 只差一条腿（Lead 亲测）
我独立调用 compile_mainline 量了三个 fixture（不是采信报告），结果与 WP1 交付逐字相符：
  copy_only      applied=(analyze,resolve,completion,storage)  stopped=apply     TL_NO_VECTOR_REGION
  independent_ub applied=(analyze,resolve,completion)          stopped=storage   UNKNOWN_CAPACITY (rmem)
  minimal_aiv    applied=(analyze,)                            stopped=resolve   PLAN_DEPENDENCE_PROOF_UNAVAILABLE
结论：**分层 gate 不是分布式死路** —— copy_only 上 plan/completion/storage 三层已由真实对象判空，只剩 emitter 的'无向量区即拒'。三个 fixture 合起来构成一个闭合三角：能发射⇒有向量区⇒需 rmem 寄存器⇒storage 因 rmem 容量 UNSOURCED 而拒。因此 ready=True 只差 **rmem 事实**这一条腿（另两条腿：carried 证明、事件注入已分别由 task-46/task-54 承担，后者已完成）。
由此调整关键路径：task-50（WP3 寄存器预算闸门）升为唯一关键路径，交 flex-4-g3（他刚证明三角并要求过 ready=True 断言）；implementer-g1 回手 task-46 carried。

## R29 方法账新增（本轮由实现者自曝，Lead 收录）
1) **第二次猜测仍然是猜测，第三次也是**：同一轮内 TILELANG_CACHE_DIR 猜两次（模块常量→EnvVar 类属性，真身是 tilelang.env 实例属性）、binary 布局、sibling 层级、三角不变量三次全错。凡'整张表的普遍规则'必须先枚举真实表再断言；最后成立的三角版本是**两个程序对照量出来的相关性**，不是推导出来的。
2) **瞬时快照 ≠ 持续状态**（第二次实例）：flex-1-g1 据 find 0 命中判 task-56 未完成，而 DRAFT.md 与他的 find 在同一分钟落盘（mtime 07:41:10）。新纪律：缺失类断言必须附观察时刻，下结论前复查 mtime/HEAD。Lead 本轮也犯过同款（05:49 把落盘竞态夸成整队冻结）。
3) **聚合相等会掩盖写法差异**：1.0 与 2**0 折叠同值，只比数值会放过被改写的表 ⇒ 校验器必须同时比对**源写法**（ast.unparse 归一后逐字段）。这条来自 task-52 实现者自测打红。
4) **命名错误的层比漏报更坏**：compile_mainline 的 REFUSED 记录原先不写 entry_identity，导致 gate 报 (entry, ENTRY_IDENTITY_MISMATCH) 而真实缺口在别处；已修为 refused 记录也写真实 entry identity，并加'没有幽灵 entry 缺口'断言。

## R30 task-52 校验器独立攻击（Lead 亲测，两个盲区一真一假）
- **真盲区**：逐个 pop 顶层字段再 validate（/tmp/test_blind2.py），7 个字段删除后**仍返回 VALID**：schema_version / registered_at_utc / operator / purpose / tree_anchor / unknowns / non_normalisation_note。其中两个非形式问题：删 unknowns ⇒ 登记物可不声明"它证不了什么"（G3 唯一诚实出口消失）；删 tree_anchor ⇒ 唯一的仓内证据（cannbench_contract.py sha256 与 :983/:1006/:1008/:934）可消失而校验仍绿。
- **假盲区（已被覆盖）**：删一条 case 至 43/44 **已被** BLOCKED_DEPENDENCY 抓住（理由是 'the distribution declares 2 supplemental_mla/bfloat16 case(s'）。但它是被分布检查**偶然**抓住，不是被 case 数/集合显式抓住 ⇒ 要求补显式断言，使失败信息指向真因，避免未来改动依赖副作用。
- 教训：flex-1-g1 的'自洽检查 0 红 / 双向重推导 2 红'在**他那个检查器**上成立，在同一契约的另一个实现上不成立（分布检查兜住了）。跨实现比较攻击面时，结论不能跨实现搬运。

## R31 task-52 guard 清单（三位探索者独立实测 + Lead 攻击，共 8 条）
1) 双向独立重推导：只做 JSON 内部自洽的检查器，对源写法与数值同步篡改的协调攻击 0 红漏检；从磁盘表重推导则 2 红抓住。
2) case 数显式断言（40 权威 + 4 supplemental = 44），id 集合必须绑外部文件（case_matrix.json / supplemental_mla.json），不能靠 JSON 内 authoritative_total（自我声明）。
3) 字段齐全性显式 guard：删字段当前是裸 KeyError；Lead 实测 **7 个顶层字段删除后仍返回 VALID**（schema_version/registered_at_utc/operator/purpose/tree_anchor/unknowns/non_normalisation_note），其中 unknowns（G3 唯一诚实出口）与 tree_anchor（唯一仓内证据）消失仍绿，属实质缺陷。
4) 比对必须走数值折叠、禁止字面串比对：ast.unparse 产出 2**(-7) 而 registry 存 2 ** -7，字符串比会把完好件判 44/220 全红（flex-1-g1 连错两版才改用折叠）。
5) 1.0 与 2 的 0 次方折叠同值，必须同时比对源写法，否则被改写的表可过关。
6) key 形态分支：40 条无 supplemental 键、4 条有 supplemental:true，统一断言会把完好件判红。
7) supplemental id 改名规则：registry 用 mla_supplemental/<suffix>，源文件用 supplemental/mla_<suffix>，不实现映射则 4/4 假红。
8) 身份 guard：comparator / threshold_table / tree_anchor 三 sha256 + line_range + repo_head 全部复算。
跨实现教训：同一契约的三个独立实现，漏检结论不能互相搬运（一位报'删一条 case 全绿'，另一实现已被分布检查偶然抓住）；但'偶然抓住'仍要补显式断言，使失败信息指向真因。

## R32 task-56 登记物自造记法（Lead 裁定升 v2）
explorer-g1 自曝并经 Lead 核源文件确认：supplemental 4 条在 registry 里写成 `mla_supplemental/<suffix>`，而源文件 supplemental_mla.json 的真实字段是 `case_id: gqa_bsnd_kv_tail` + `key: supplemental/mla_gqa_bsnd_kv_tail` —— **源文件两个字段都不是该记法**，是生成器把 `supplemental/` 与 `mla_` 重拼出来的自造值。权威 40 条无此问题（`<case>/<case_id>` 与源直接相等）。
裁定 **升 v2**：cases[] 新增 `source_key`（逐字拷贝源文件 key，40 条也加）、`source_case_id`、`source_file`。这样校验器**不需要任何改名规则**，可直接断言 source_key/source_case_id 集合与源文件相等。保留 v1 的 sha256 与 mtime 一行作为历史（883117e），draft 仍 true、时间戳标注 v2。
裁定理由（属方法账）：把登记物缺陷转嫁给校验器，等于让一条不可验证的私有改名规则成为中间层——它错时要么安静放行要么误杀，而它已经误杀过一次（4/4 假红）。这是同一错误形状的第 N 次出现：**自造值 + 下游规则**。
另一条采纳：**同一契约的不同实现，漏检结论不能跨实现搬运**（一位探索者报『删一条 case 全绿』，另一实现已被分布检查偶然抓住）。但『偶然抓住』仍要补显式断言，使失败信息指向真因。

## R33 Lead 撤回一条裁定：纯搬运 kernel 是真实程序形状（register budget 正例改道）
起因：flex-4-g3 诚实报告 task-50 拿不到'可用寄存器数量'的真实数字，并指出真来源可能是编译期 JITKernel.n_regs。我核实并作出两处更正：
1) **编译期资源采集在此 build 的 ascend 上不存在**：tilelang/jit/kernel.py:264 `capture_hip_resource_usage = is_hip_target(target)` —— 只有 HIP 目标接资源采集。故 AIBIN_PROBE 的 n_regs/n_spills=None 不是暂缺，而是该 build 不提供。
2) **我撤回'不许写纯 copy 渲染路径'的裁定**。当时理由是'不能为绕开 rmem 制造不真实的程序形状'，这是错的：DMA-only kernel（GM→UB→GM）本来就是合法的 Ascend 程序，copy_only 的 TL_NO_VECTOR_REGION 是**发射面缺口**而非事实缺口。故改为要求实现纯搬运支持，且文档措辞必须是'emitter 支持无向量区的纯搬运程序'、不得写成'为绕开 rmem 而加'。
处置：
- **task-50 按现状收尾**：闸门 + 双向事实面 + 负例（STORAGE_REGISTER_UNSOURCED / OVER_BUDGET / 双侧齐备才放行）为终态；'可用数量无真实来源'是**事实**而非待办，须在 memory.REGISTER_BUDGETS 注释写明并标注所有寄存器域 UNSOURCED，避免后人以为漏做。
- 寄存器预算**不追求正例**（无来源即无正例），report 侧（lowering 按生命周期报占用）保留但标注'单侧可测、另侧 UNSOURCED ⇒ 仍拒'。
- 正例改道：first ready=True 应来自 **纯搬运 kernel**（不依赖寄存器预算事实）。

## R34 方法账第 9 条（由实现者九次同形错误归纳，Lead 采纳）
**凡'某侧沉默'，先问它该不该说话，再判定是缺项还是通过。**
今日九次同形错误：把对象当成它的字符串（`str(OwnerKey)` 匹配不上任何预算键）、把宽度当成数量（256B vreg 宽度 ≠ 每域可用数量）、把『这一侧没说话』当成『没有要求』（程序不使用的域被报缺、预算缺失被当通过）。修法都不是再猜一次，而是读真实对象/枚举真实表。

## R35 v2 落法定稿 + Lead 指令更正（会误杀 44 条）
我先前指示 v2 的 `source_key` 存 `case_matrix.json` 的 `key` 字段 —— **错误**，实测会误杀全部 44 条：源条目字段是 `case=attention_residual / case_id=1 / key=level4/attention_residual_1`，而 registry 40 条 id 是 `attention_residual/1`（= `<case>/<case_id>` 拼接，**不等于** `key`）。这条更正由 flex-1-g1 交叉核实触发。
定稿（附加式 C 组，不改现有 case_id）：
- 新增 `source_key`（逐字拷贝源 `key`：40 条形如 `level4/attention_residual_1`、4 条形如 `supplemental/mla_gqa_bsnd_kv_tail`）、`source_case_id`（源 `case_id` 原文）、`source_file`。
- 契约只有一条：`source_key` 等于源字段原文；其形态（`level4/` vs `supplemental/mla_`）**不构成语义**，不许再出现统一规则或第三种自造记法。
- 校验器同批落盘：删掉 suffix 映射与 `prefix == "mla_supplemental"` 断言（flex-1-g1 实测该断言会对『把 case_id 改成源字面』误杀），改为 source_key/source_case_id 集合逐字相等 + dtype 逐条相等 + 一条反向断言（case_id 是 registry 记录 id，但 source_key 必须被源字段支撑）。
- 保留 v1 sha256/mtime 一行作历史，draft 仍 true，时间戳标 v2。
方法教训（新增）：**把缺陷转嫁给下游规则后，修正上游反而先红** —— suffix 映射规则存在时，把 case_id 改成源字面会先撞规则；这使'修复'看起来像'破坏'。私有改名规则不仅有误杀风险，还会**阻止上游被修**。

## R36 v2 独立复算通过 + 校验器一处装饰性字段
flex-1-g1 独立复算（不采信自述、不写 registry）：registry sha 4a77755e… / 52065B / schema v2 / 44 条 / draft=true，复算期间 mtime 未变。结论：
- 折叠：从 cann_bench_compare.py 独立 BinOp-aware 重推导，44 条×5 字段×两侧 = **440 检查 0 不符**；dtype_entries 7 行相等；float8_e5m2 末两档仍是字面 1.0（未归一化）。
- 分布：AR bf16×17+fp16×3 / MLA fp16×15+bf16×5 / suppl bf16×2+fp16×2 = 44。
- 无通过声明：PASS/通过/数值通过/VERIFIED/certified/DEVICE_VALIDATED 命中全 0；唯一 passed 在否定句。
- source_key 44 条**全局互异**且逐字等于源 key（40 条 == case_matrix key 集合、4 条 == supplemental key 集合，按各自 source_file 对号 0 例外）；source_case_id 仅 24 distinct（40 int 1..20 重复 + 4 str）⇒ **全局集合断言必错**，per-operator 唯一性成立。
- v1 历史经 `git show 883117e` 实际字节 sha256 = b25dd485…，与 registry_history 记录相等；v2 的 44 个 case_id 与 v1 逐位相同（未重写历史 id）。
**残留（真缺陷，已派修）**：篡改 `source_file` 仍返回 VALID —— validate() 只用它做分类，真正读盘走模块级硬编码常量（:413-414），该字段目前是装饰性的。修法二选一：(a) 按每行 source_file 分组读盘；(b) 在 binding_rules 明写其为注记并由校验器断言与固定常量一致。要求不留被忽略的字段（'被忽略'必须变成'被核验'）。
方法观察：flex-1-g1 期间三次撞上半落盘状态（调用未定义函数、sha 连变 4 次），他没有在抖动期下结论，而是等 20s 内 sha 不变才跑正式套件 —— 这是'瞬时快照 ≠ 持续状态'那条纪律第一次被**主动**执行而不是事后追认。

## R37 环境事实更正：sourced CANN 下 import tilelang 会带入 torch_npu（但不碰计算节点）
此前多处结论写的是「未 import torch_npu」，**该结论只在 bare 环境下成立**。实测（source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh 后）：
- `import tilelang` 会**连带加载 torch_npu**（sys.modules 中存在）；bare 环境不加载。
- 因此在 test_tl_emit_read 里断言 `'torch_npu' not in sys.modules` 是**环境相关**断言：sourced 下必红，bare 下绿。
- 关键实测（/tmp/probe_fd.py，sourced 下跑完整链到 compile 成功，compiled=True）：对 /proc/self/fd 做 readlink 集合差，被打开的只有 `['/dev/davinci_manager', '/dev/hisi_hdc']` —— **没有** `/dev/davinci<digits>` 计算节点。
结论与 R11 已定规则一致：控制节点由 import tilelang 被动打开 ⇒ 只记录；计算节点 ⇒ 拒绝。故该测试应改为断计算节点未打开（AIBIN_PROBE 的 fd 取证法），torch_npu 出现只记录并注明环境，另保留「无 kernel launch / 无 npu 张量操作」这类与环境无关的断言。
另一个数字事实（sourced）：全量 299 passed / 1 failed（仅上述环境相关断言）；bare 下 300 passed。

## R38 gate 只剩三条，且都是无条件拒绝的残余
DMA-only 支持落地后，copy_only 跑完整个事务（applied=analyze,resolve,completion,storage,apply,verify；reader VERIFIED），gate 缺项从 10 条降到 **3 条**：
- chain:TRANSACTION_INCOMPLETE（同批事务尚未收全）；
- event:TL_EVENT_LOWERING_UNOBSERVED（需 apply 注入事件后由 reader 观察）；
- install:INSTALL_CANDIDATE_UNVERIFIED（来自 `plan/mainline.py:187` 的**无条件** INSTALL_TL_CONFORMANCE_NOT_IMPLEMENTED，属 R7 形状第四处）。
裁定：临时授权 flex-4-g3 只改 `plan/mainline.py` 一个文件，把该处改成逐层条件判定（receipt / VERIFIED verdict / ABI / 候选对象同一性，缺哪层点名哪层），**不得删掉 TL conformance 这一层**，只把它从「永不满足」变成「可被真实对象满足」。

## R39 方法账第 9 条定稿（由实现者 10 次同形错误归纳）
**凡「某侧沉默」或「某条恒等式」，先枚举真实样本再断言；恒等式尤其危险，因为它看起来像定义，实际是假设。**
10 次实例：TILELANG_CACHE_DIR 猜两次错、二进制目录布局、sibling 层级两次、三角不变量三次、OwnerKey 当字符串、__CCE_KernelArgSize==参数数×8（实测 2 参与 3 参都报 24，恒等式被证伪）。
