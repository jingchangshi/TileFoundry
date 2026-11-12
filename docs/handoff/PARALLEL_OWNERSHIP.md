# 并行作业硬边界（A–F 所有权）

基线：当前工作树。生产文件一次只有一个活跃写者；测试文件与其被测生产文件归同一包。跨包只通过已冻结字段/函数接口交互，不直接编辑对方文件。identity与delays为共享只读合同；需改时停相关包，由单一写者同批修改生产与全部消费测试。

| 包/任务 | 唯一写者范围 | 只读输入/接口 | 独立验收 |
|---|---|---|---|
| A WP1 S0 positive gate / task-53 | 新增gate正例测试；p3_chain、plan/mainline、passes/gate_layers的最小事务接线 | plan/completion/storage/emit/read/compile产物接口 | 同一真实对象链使require_export_ready返回ready=True且gaps=[]；篡改任一对象或stored identity后具名变红 |
| B WP2 apply事件注入 / task-54 | plan/apply.py、plan/verify.py、completion/bind.py及各自测试 | ResolvedPlan、CompletionPlan、source_map、现有事件op/TL surface | independent_ub候选出现真实FlagSet/FlagWait；source_map含SYNC角色；verify双向核对；删/改端点必失败 |
| C WP3寄存器预算 / task-50 | memory.py、tf_target.py、storage/__init__.py及register测试 | Target facts、StoragePlan、lowering占用报告接口 | 双侧事实齐备且占用≤可用时rmem程序storage正向；任一UNSOURCED或超预算分别返回指定拒绝 |
| D WP4 carried soundness / task-46 | plan/problem.py、plan/source_audit.py、plan/resolved.py及carried测试 | UnitTree、RegionBindingSummary、AccessRelation、delay三态 | 每条跨迭代root的RW关系成边或具名拒绝；step/binding/读写次序可判；UNSOURCED read-complete不得用occupancy或latency_model冒充 |
| E WP5 BF16结构面 / task-51 | codegen/emit.py、vector_model.py、tl_reader.py、ir/isa.py必要契约及bf16测试 | 真实bf16作者程序、现有carrier/convert op语义 | bf16程序emit/parse/read结构一致；16-bit lane provenance为UNSOURCED，状态INCOMPLETE，不冒充VERIFIED或数值通过 |
| F WP6比较器登记 / task-52 | COMPARATOR_THRESHOLD_REGISTRY.json及其只读校验脚本/报告 | npu_compare、CANN_BENCH_THRESHOLDS、case dtype分布 | 每case登记路径/sha256/version/accuracy_mode/threshold_entry；缺字段或sha变化即BLOCKED_DEPENDENCY，不改比较器/proto |

必须串行的三处接缝：
1. B产生候选与receipt → A安装/gate正例；A不得自造事件或绕过verify。
2. C产生storage/寄存器事实 → B/A消费；storage身份变化后B/A必须重跑。
3. D改变problem/distance/delay → C/B/A全部下游重算；不得只更新stored identity。

落盘纪律：一次write/edit完成一个可解析单元；宣布前运行compileall、import及本包测试。跨分钟collection error先复测再告警。禁止用新增负例替代正向真实对象链；每包至少给一个正例或明确BLOCKED_ENV。
