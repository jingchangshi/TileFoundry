# TL-DTYPE-SUPPORT / IDENTITY-BYTES（task-42 第 3 条 + lead 四条判据）

owner flex-4-g3。全量 201 passed（test_completion 11、test_tl_emit_read 23）。
这一档记三件事：支持域扩到哪为止、lane provenance 凭什么判定、身份字节怎么防漂移。
我自己被推翻的地方留原文。相关代码：codegen/vector_model.py、codegen/tl_reader.py、
completion/plan.py；测试名逐个写在下面。

## 1) 支持域如实上报：16-bit 可发射不可认证，且目前没有端到端路径

把 bf16/f16 加进 emit 的 ELEMENT_DTYPES **确实扩大了发射支持域**，但层要分清：

- 第 0 层（作者前端）先拒，且不在本切片代码里。把 minimal_aiv 的 dtype 换成 bf16，
  checked_program/analyze 阶段就失败；实测栈在 src/tilefoundry/ir/hir/schedule.py:342
  （_infer_schedule），异常类型 VerifyError，且没有 .code 属性——即它不是我们具名拒绝，
  是 DSL 契约本身说不支持。原文：
  「ScheduleOp: T.ascend_vld ... src.dtype=bf16 dst.dtype=bf16, reads bf16 loads state
  dist=UNPK_B16 and land in the carrier class; f32 loads state no dist and land in an
  element register」。f16 更早失败（同目录 :40 variable 'carried'，src: f16 does not match f32）。
  这正是 lead 定的 G1 三项缺：carrier 载入 transport、f16↔f32 双向转换、f16 输出存储。
- 第 1 层（本切片 reader）：给定 16-bit 结构，lane 模型 UNSOURCED → verdict 上限
  INCOMPLETE，verified 必 False。

结论：G1/G2 未闭合。16-bit 目前既走不到 emit 也走不到 reader；reader 的 16-bit 处理只是
把「将来进来了也不许认证」这道闸先立住。
test_sixteen_bit_has_no_end_to_end_path_and_is_not_reported_as_one 钉住第 0 层的异常类型、
抛出模块与契约关键词，防止这段被读成端到端支持。另：copy_only（纯 GM→UB→GM）在支持域外，
由 TL_NO_VECTOR_REGION 具名拒（test_copy_only_is_outside_the_vector_support_domain_by_name）。

## 2) lane provenance 是算出来的；判定读结果，不读标签

codegen/vector_model.py 里 VectorModel 不接受 provenance 参数，它由 _cross_check 计算：

    holds = (lanes, element_bits) ∈ target 声明几何  AND  lanes * element_bits == 2048

declared_geometry() 从 ir/isa.py:39-41 实测只返回 [(64, 32)]。element_widths=(16, 32) 只声明
「这两种宽度可做」，并不同时声明 16-bit 的 lane 数——这个区别就是下面结论的来源。
REGISTER_FILE_BITS=2048 与 memory.py:85-86 声明的 256 字节向量寄存器一致。

| element | lanes | vector dtype | holds | basis | 依据 |
|---|---|---|---|---|---|
| float32 | 64 | float32x64 | True | GEOMETRY | 2048//32=64 恰等声明 lane_count，且 64*32=2048 是整条寄存器 |
| bfloat16 | 128 | bfloat16x128 | False | GEOMETRY | 未声明 (128,16)；声明的 64 lanes@16bit=1024 只是半条 |
| float16 | 128 | float16x128 | False | GEOMETRY | 同上 |
| float8_e5m2 | 256 | float8_e5m2x256 | False | GEOMETRY | 未声明 (256,8) |
| bool | 256 | boolx256 | True | DEPLOYED_SURFACE_CONSTANT | 依据是部署面常量 pset→boolx256（simd.py:105-111），不是 2048//1 |

所以 32-bit 的 SOURCED 不是默认信任 fp32（等式两半都写成显式断言，
test_thirty_two_bit_lane_is_sourced_by_the_cross_check_not_by_default）；
16-bit 的 UNSOURCED 也不是我们不信 fp16，而是缺一条 target 声明
（test_sixteen_bit_is_unsourced_because_the_target_declares_no_such_geometry）。

tl_reader.lane_provenance_gap 读 cross_check 的 holds，产出 gap（不是 refusal），payload 全是
机器值（element, lanes, vector_dtype, declared_geometry, computed_lanes, register_file_bits,
whole_register）。三态因此是：结构不一致 → REFUSED；结构一致但 lane 无据或事件层未观测 →
INCOMPLETE；全部有据且一致 → VERIFIED。

判定确实只读那个字段，反证写法（lead 第 2 条要求，且我按你的警告把它绑到交叉核对结果而非
可改标签）——test_lane_provenance_reads_the_cross_check_not_a_prose_label：
- 代入 provenance=SOURCED 但 holds=False 的模型 → gap 仍在；
- 代入 provenance=UNSOURCED 但 holds=True → gap 消失；
- 把 target 声明几何换成 ((128,16),)（即 32-bit 不再被声明）→ expected/observed 完全一致、
  零 refusal，verdict 仍从可能 VERIFIED 掉到 INCOMPLETE
  （test_unsourced_lane_model_caps_the_verdict_at_incomplete）。这说明依据是声明，不是名单。

自我推翻留档：这条反证最初是假通过的。我写的是 replace(model, provenance=...) if False else
model —— 一个空表达式，什么都没测；且第一版还多写了一句 provenance != SOURCED 的假断言。
重写为真代入并加 holds 双向翻转后才有意义。32-bit 三档 fixture 未退化：
test_thirty_two_bit_fixtures_are_unaffected_by_the_lane_gap（refusals 空、gaps 只剩事件层）。

## 3) 身份字节防漂移：声明式白名单，不是散文黑名单

completion/plan.py：_declared_leaves 先枚举事实载荷的全部叶子路径（真实 facts 上实测 83 条），
PROSE_PATHS 只显式声明 3 条为散文，其余 80 条全部参与哈希。方向是 fail-safe：给
EventPool/HardEventPoolFacts/TargetEventFacts 加字段自动进身份，不需要改任何清单；要让某句
话不进身份，必须显式登记它的声明路径。test_new_model_field_participates_in_identity_by_default
断言 PROSE_PATHS ⊆ 实测叶子（防陈旧条目掩盖字段改名）并注入全新字段确认 identity 改变。

同一档钉住 semantic_identity 与 build_identity 的分界：内容摘要（binary_sha256、
host_library_sha256、ascend_source_sha256、machine、elf_type、argument_size、manifest 文件表）
属语义；compiled_at、cache_directory、toolchain_string 属本次构建。
这里又抓到一处真缺陷：semantic_identity 原先不含 binary_sha256——因为 kernel manifest 根本不
列 aibin（它在另一个不可变目录），于是「host 侧清单相同但 aibin 不同」的两个构建会被宣称为
同一程序。是 mutate 测出来的（改 binary_sha256 而 semantic 不动），现已补齐三个内容摘要并
各自断言（test_every_identity_producer_is_audited_for_prose_participation）。

我在第 3 条路上踩的两个 bug（都被自己的探针当场推翻，留原文）：
1) prune 写在 digest() 之后：散文已经进了哈希，剥壳是空操作 → reword-invariance 实测 False。
   改为先剥后哈希。
2) 路径拼接带前导点（".event_facts.pools[].source"），与 PROSE_PATHS 里的
   "event_facts.pools[].source" 永远匹配不上 → 同样什么都没剥，也是 False。
   统一 grammar 后双向才成立：reword-invariant True / capacity moves True。

## 4) 伪造类测试同型自查（lead 的全链规则）

写了 pytest 插件 /tmp/noop_plugin.py 扫全仓；它自己的两次假清白就是这条规则的教材：
- 第一版在 import 时读 sys.argv 找测试文件，而 pytest 已改写 argv → 扫到 0 条却打印
  「none」，看起来像全部干净。改为从 collected item 取文件，并在报告里印 examined 条数。
- 第二版用 obj == result 判空操作，但 ResolvedPlan 的 problem_ref/target_ref/dependence_proof
  是 compare=False，于是 replace(plan, problem_ref=None) 与原来「相等」却确实是真改动 →
  会把真篡改误报成空操作；更重要的是，任何依赖 == 的门同样看不见这类改动。最终版按字段名
  逐一比（值 + 对象身份），并单独报出「真改动但 == 看不见」这一类。

结果（examined 82 条含 pytest.raises 的测试）：
- 我的文件无真空操作。test_effects_must_be_the_plan_problems_own_records 是有意保持
  value-equal 的篡改（它测的是记录身份，不是内容），已把这点写成显式断言：
  new is not old 且 new == old，并注明「若不满足就是在为错误的理由拒绝」。
- 授权链另有正向证明（test_equivalent_content_is_consumable_and_changed_content_is_not）：
  等价 twin 可消费；把某 channel 的 versions 改 1 后，先断言内容 identity 确已改变再断言被拒；
  该篡改内容重新 verify 也拿不到授权。我最初那版「伪造 receipt」测试就是因为没重算
  identity，内容与注册表里那条一致，其实是假通过——已扩成三段。
- 别人文件里 5 处候选（未必是缺陷，可能只是我的启发式过宽，我不擅动别人的文件）：
  test_numeric.py:281 AccessFact(generation)；test_resolved.py:386 BufferPlan(buffer_ref)、
  :394 Region(region_id)、:395 ResolvedPlan(regions)、:396 RegionPlan(region_ref)。

## 5) 注册表按内容键，判据写明（lead 第 2 条：要理由不要断言）

保留内容相等，不改成 is。require_completion_verified 的 docstring 写了完整判据：查 key →
比内容 → 用保存下来的真 plan/target 重跑 _verify_structure 与 _correspondence → 比
recheck.identity。它比 is 更稳：跨进程重材料化与 dataclasses.replace 的等价副本都不该误拒，
而能造出等价对象的人本来也能造出 identity；残余风险只剩 digest 碰撞（截断到 64 位，属哈希
碰撞域）。要更强应收长 digest 并重哈希既有身份，而不是加 is。

## 6) task-40 step2 的两个结构性前置（等 lead/architect 定）

1) _Emitter._contract 目前按「类名去掉 ascend_ 前缀」查 ISA。事件 op 类名是
   FlagSet/FlagWait/Drain，不以 ascend_ 开头，实测直接吐 TL_INSTRUCTION_UNREGISTERED。修法
   明确：改用 implementer 已有的 event_contract_of()（它按 _op_schema.name 查并拒绝把数据
   指令当事件）。这条在我范围内。
2) 关键冲突：flag IR 的 attribute schema 是 int（slot/generation/versions/flag_id），且
   _check_common 校验 slot == generation % versions；而 completion 是有限符号模板
   （generation 是 Expr，尾部是 max(0,N-V)..N）。按 int 逐代实例化会让事件条数随 N 增长，
   直接违反 lead 钉的「不随 N 增长」。后端 T.ascend_set_flag/ascend_wait_flag 的 event_id
   文档明写 int or PrimExpr，所以符号 event_id 本身可发。三条路：
   (a) 放宽 src/tilefoundry/ir/tir/flag.py 让 generation/slot 收 Expr，并符号地证
       slot == generation % versions（改 core IR，超出我的写范围）；
   (b) 事件语句留在控制 envelope 内（if in_domain(k): set/wait），k 是循环变量，一条语句
       覆盖全部代次 —— 我倾向以此为主；
   (c) 新造一套只承载 Expr 的事件 op —— 我反对，那是第二套事件模型。
   定下来之前我不写 step2 的 renderer：(a)/(c) 决定 op 形状，(b) 决定要不要它。
