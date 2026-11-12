# TL-EVENT-SURFACE（task-40 step2 + task-48）— 事件发射面与 EXIT 义务存在性
 实测：事件发射面（S0 具体 int）

权威与限制（lead 要求留档，勿删）：
- hard-event 串的唯一权威是部署 build 的两处 C++：GetResourcePipeName
  (tilelang/src/ascend/transform/ascend_pipe.h:75-91，命名 MTE1/MTE2/MTE3/M/V/FIX/S)
  与 GetHardEvent (insert_sync.cc:751-765，成对拼接后再施加四条别名
  MTE2_M→MTE2_MTE1、M_MTE2→MTE1_MTE2、MTE3_M→MTE3_MTE1、M_MTE3→MTE1_MTE3)。
- **Python 侧不校验这个串**：ascend/language/sync.py:44-45 只检查 isinstance(hard_event, str)。
  所以名字错了不会有任何一层替我们拦住，错误会一路到 pto.set_flag 的实参
  (codegen_pto.cc:4847-4853)。这就是为什么 emit 侧照抄 C++ 表并注明来源。
- 实测发出：T.ascend_set_flag('MTE2_V', 3) / T.ascend_wait_flag('MTE2_V', 4)；
  parse 成功；read_tl_facts 读回 (('literal','MTE2_V'),('literal',3))，
  unsupported 由 (('op','tl.ascend_set_flag'),...) 变为空 —— 即"认得并核对"，不是"没见过"。
- 一条语句进、一条语句出：3 条义务在 N=1/2/4/8/64 下恒为 3 条事件语句（渲染器不循环）。
- drain → TL_EVENT_SURFACE_MISSING（部署 sync.py 的 __all__ 无 drain intrinsic）；
  pipe_barrier/threadfence → TL_EVENT_TF_OP_MISSING（TF op 不存在，且它们不排序 flag 所排序的东西，
  不得替代 drain）；跨核 → TL_EVENT_CROSS_CORE_UNSUPPORTED（bank/mode 声明为 UNRESOLVED）。
- 具体 int 按 lead 裁定：非具体 generation → TL_EVENT_GENERATION_SYMBOLIC
  （实测 flag IR 确实接受 Var/DimVar，Constant 折叠为 int，所以这条拒绝是可达的，不是摆设）；
  slot != generation % versions → TL_EVENT_SLOT_INCONSISTENT；
  flag id 越出 target 声明的 intra-core 池（AIV=8）→ TL_EVENT_SLOT_OUT_OF_POOL。

## 2) task-48：EXIT 义务存在性 —— 四种删除形态的最终码名（independent_ub 真 plan，9 sites / 5 channels）：
(a) 只删通道 → COMPLETION_SITE_UNBALANCED
(b) 通道+两 site+exit_site_ids 三者一致删除 → COMPLETION_OBLIGATION_MISSING（原为 ACCEPTED，已闭合）
(c) 删通道+两 site、exit_site_ids 留旧 → COMPLETION_EXIT_SITES_MISMATCH
(d) producer 指向不存在 site → COMPLETION_CHANNEL_ENDPOINT_UNKNOWN
(b) 是 site-balance 与 exit_site_ids 一致性的共同盲区（两者都是一致性检查，不是义务存在性检查）；
现由 bind.exit_obligations() 从 problem 记录独立推导 ("EXIT", region, task)
（WRITE_VISIBLE 的 GMEM 边界写且区域内无读者），与 builder 的 _drains() 互不复用，
_correspondence 双向核对。码名沿用既有 COMPLETION_OBLIGATION_MISSING，未造同义码。
未闭合（如实）：跨层回归"apply 注入的事件被删 → verify_actual_program 失败"归 task-46 作者，
我做 reader 侧对手测试，同批落盘。

自我推翻与假清白（留原文，方法条目用）：
1) "16-bit 没有任何作者程序能走到发射"——错。我用 minimal_aiv 文本替换 f32→bf16 得出的结论，
   那产出的是**非法**程序，死在 TileFoundry 自己的 HIR 校验器 (ir/hir/schedule.py:342,
   VerifyError 且无 .code)。lead R10/R12 的写法才对：**正确作者语法**的 bf16
   （carrier tile + ascend_vld dist=UNPK_B16 + ascend_pset16 + ascend_vcvt 双向 +
   ascend_vsts pack=PK_B32）能 FINALIZE 并**确实走到我的发射支持域**，在那里被
   TL_BUFFER_SCOPE_MISMATCH（carrier scope 未承认）拒。已改为两条分别断言的测试。
2) 静态守卫第一版按行文本 grep，"发现"违规就在本模块自己的解释注释里——把提到某模式的
   散文当成了使用该模式。改为 AST，只检查真实调用；并用注错代码证明它真的会失败
   （注入 .upper() 后测试 FAILED 并报出 completion/verify.py:663）。
3) _event_pool 曾写 hard_event_pool(core.upper())，而 target 声明 AIV="aiv"
   (completion/model.py:32) → 查询返回 None。若 None 被读作"无上限"，**所有** flag id
   都会静默通过容量检查。现在只经 declared_core() 解析，且 None 一律具名拒绝。
4) 我的"池查询不许手拼大小写"测试原先断言 actual 里不许出现 EXIT —— 错：幸存的那条 EXIT
   本来就合法地出现在已保护集合里。改为断言缺失的那条不在、幸存的那条在。
5) 落盘事故（全队冻住两次）：同一脚本里先插入常量、再用**插入前算出的行号**切片 →
   把上一个函数的 payload = { 截成半截，整目录 collection error。教训与 lead 方法条目 7
   同类：脚本凡"先改后定位"必须重读文件再算边界；现在每次落盘后先跑
   compileall -q targets/ascend src/tilefoundry 再报数。
