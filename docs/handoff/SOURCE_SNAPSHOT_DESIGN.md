# 规范化来源快照、真实问题绑定与计划内容身份

设计基线：HEAD f341e47529453bdbc8dd81f2c108d36628fb4b8b，读取时包含未提交的原生plan/source_audit实现。本文仅为设计报告；不修改生产代码、不执行设备或把设计当验收。
本规范演进既有SourceSnapshot、NormalizedProblem、ResolvedPlan和identity函数，不定义另一份计划、completion schema或通用序列化框架。规范字节是这些记录的确定性视图，不是可执行IR。

## 1. 当前可复用部分与真实缺口

| 位置 | 已有能力 | 本次需要补足 |
|---|---|---|
| [identity:107](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/identity.py:107>) | 读取真实Op的attribute参数；:127区分参数/Let/For/Mesh binder；:177含output_count | :78未绑定Var退回名字；:127全树预绑定不检查词法可见性；:89 Var分支漏局部Var自身type；:71仅str(type)；:124任意object repr；:98按repr排序map；:158 target只class/name |
| [SourceSnapshot:175](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/units.py:175>) | 已有module_ref、entry_ref、canonical_bytes、identity | :937字节来自上述旧entry payload，只保留入口；缺完整Module环境/目标事实、闭合callee及受限值域；不能说“字节字段不存在” |
| [problem identity:819](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/problem.py:819>) | 有coverage与task/access/effect来源ID | 当前hash仅四类ID列表，不含relation参数、guard、edge/reuse、资源offset、容量；:212 target repr不是事实快照 |
| [ResolvedPlan:70](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/resolved.py:70>) | 唯一计划、真实Region/Task/Buffer refs与时序 | 无problem_ref；:377 region hash漏bounds、domain、来源refs；:398计划hash不绑定完整原问题 |
| [source audit:25](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/source_audit.py:25>) | 从实际指令读operand身份、MemoryEffect、pipe与completion；:127真实UB区间；:176单region reaching-def | :28明确原检查不证明alias/guard/generation；:274依赖审计仍因delay未证明终止。身份完整不替代独立source audit，不能因此移除该拒绝 |

## 2. 一个编码入口，四个内容边界

在现有identity模块提供source_bytes、target_bytes、problem_bytes、plan_bytes四个明确入口，内部共用受支持类型的编码函数；已有function_identity/module_identity改用同一基础编码，不保留旧repr备用路径。编码不是新增编译driver。
固定UTF-8 JSON、sort_keys=True、separators=(',',':')、ensure_ascii=True、allow_nan=False；每个节点与值带类型标签。整数用十进制串，bool独立标签，float以IEEE754 binary64大端hex编码（含−0/inf/NaN位模式），bytes用hex，None独立标签。带域标签的payload含规范编码版本常量，用于失效而非运行时版本分派。
tuple/list保留有语义的原顺序；set/frozenset按元素规范字节排序；普通map先编码key，再按key字节排序，禁止repr排序。AccessRelation例外：relation按(dim_type,index,coefficient)位置编码，values按参数名字排序位置绑定真实Expr，不把ISL文本或名字字符串当语义内容。相同规范key对应不同实体时拒绝歧义，不静默覆盖。所有不支持的值返回IDENTITY_UNSUPPORTED_VALUE并含字段路径，不用default=str。

| 边界 | 精确包含的内容 | 身份与排除项 |
|---|---|---|
| Target | 精确target注册类型/名称、当前实例aiv_count/aic_count等公开语义参数；当前Target查询的TopologyFacts、MemoryExecutionFacts、MemoryHierarchyFacts、space/transfer/layout/event facts与所用ISA/资源/完成cost profile | target_identity=H(target_bytes)；每事实value/source/status与UNKNOWN显式进入；不hash方法对象、repr、地址或设备探测副作用。先固定当前profile所需查询集合，新增机制显式扩展该集合 |
| Source | 实际lowered Module.entry、目标/拓扑环境、入口签名/ABI角色、完整入口body、可达原生callee闭包及Target字节 | SourceSnapshot.canonical_bytes=source_bytes，identity=H(bytes)；入口hash与全Module hash仍分别可用，不能拿二者互代；不可达函数在安装边界另以module_identity保全，不因未参与source闭包便允许改写 |
| Problem | source快照身份、target bytes、model_version、全部UnitTree来源关联、RegionProblem内容及coverage集合 | problem_identity=H(problem_bytes)；包含下节全部字段，排除identity字段本身及coverage.problem_identity以免递归；coverage.source_identity须等于固定source身份 |
| Plan | 固定source/target/problem身份与全部已决RegionPlan/TaskPlan/BufferPlan、task_to_native_source、feasibility、optimality及model_version/iteration（若已有） | plan_identity=H(plan_bytes)；排除plan_identity/region.identity本身与后续completion/storage/map/verdict/本地live refs；region.identity独立重算其真实内容，不只信缓存字符串 |

SourceSnapshot.identity的内容边界按上述明确升级；所有源消费者统一使用它，不再假定它等于旧function_identity(entry)。入口identity仍单独用于NativeVerdict/安装ABI核对，不复制另一份源权威。

## 3. Source字节必须从真实IR读取

语句按实际词法occurrence分配s0/s1…，binder按参数次序与定义位置分配b0/b1…；本地变量名不作为定义身份，公开callee/entry symbol保留。Let.value在绑定新Var之前编码；For bounds、Mesh表达式在外环境编码；新binder只对body可见；If两臂独立环境，退出恢复。悬空Var拒绝，不退回Var.name。
每个binder定义同时编码完整type/annotation一致性；每个Var use编码binder ID与实际携带type。外部已声明参数、合法capture须可沿binding summary映射，不能自动当free变量。表达式DAG共享定义有稳定引用表；statement多重出现按occurrence保存，不被id字典去重。

| IR内容 | 规范字段与支持边界 |
|---|---|
| PrimFunction / Module | ordered params的定义/type、output_count、实际target/拓扑、entry symbol、实际callee symbol与签名；不遗漏非入口可达函数 |
| Sequential / For / If / Let / ExecutionScope / MeshScope / Evaluate | 实际节点种类和字段、For IV及lower/stop/step/pipeline、If两臂guard、Let值/定义、execution kind、mesh形状与binding、callable及全部operand次序；While/Return等未实现语义可具名拒绝，不能跳过 |
| Op / Constant / Call | 必须是注册op，以完整dialect/category/name与有序attribute列表编码；所有operand与type递归编码。Op.params失败是拒绝，不能吞异常当无属性；合法纯维度Op也按显式种类登记 |
| type / layout / storage | TensorType的shape、DType所有语义位宽/符号/指数尾数字段、StorageKind枚举值、layout种类/shape/strides；TupleType/UnitType/PointerType逐已支持字段编码。Layout/Swizzle/ComposedLayout/ShardLayout只在已覆盖字段与Expr引用时支持，否则拒绝；不靠str(type) |
| AllocTensor / TensorView | allocation定义ID、实际type、view base定义或地址常量、byte offset、shape/dtype/layout/stride；静态支持域额外读取精确[start,end)区间与别名组，同地址相同区间归一，部分重叠由source audit拒绝；未知区间不填0 |
| PipelineSpec / source绑定 | 作者max_stage/max_buffer_versions/fixed_ii/enable_offset、TaskIntent/BufferSchedule的全部语义域；anchor/root/source refs编码为真实occurrence/binder定位，不照旧_SKIP_FIELDS整段忽略。诊断origin/metadata可忽略，但RegionPlan.origin是语义整数必须入plan bytes |

当前首个可实现域可先限已证明单region静态64-f32连续tile、合法mask、原生注册copy/VF指令；source_audit的限制仍独立生效。不支持一般动态layout或cross-region不意味着基础encoder可以悄悄略去这些节点。

## 4. Problem内容与实际引用

NormalizedProblem不另包一套schema。resolve_schedule保留实际输入problem为ResolvedPlan.problem_ref（必需、compare=False/repr=False），新增problem_identity、problem_canonical_bytes、plan_canonical_bytes三个固定值；SourceSnapshot原有字节继续复用。problem保留实际target_ref（同样本地字段），供没有target参数的apply/require重读，不能只保存target_identity字符串。
每个RegionPlan.region_ref必须is该problem中对应RegionProblem.region_ref；每TaskPlan.task_ref、BufferPlan.buffer_ref与来源对象对应。集合ID完整一一映射，无重复/foreign/source切片；不能仅按region_id字符串相等。对象身份比较与内容比较同时成立。

| 记录集合 | problem_bytes逐项必须读取 |
|---|---|
| UnitTree / region | 实际entry/source snapshot关联、父子顺序、真实unit node/body occurrence、captures/carries/yields、guard、intent、region bounds/activation、binding对应、task顺序；引用回source定义表，不递归live对象环 |
| tasks / uses / resources | task_id、unit真实引用、source_order/authored_stage/latency；每use的ResourceKey(完整OwnerKey/core/pipe/instance)、duration、**offset**；资源capacity/source/status；owners/pipes完整集合 |
| buffers | ID、真实root/allocation定位、space/OwnerKey/dtype/shape/layout、Fixed/Auto完整域、generation_delta、slot_bytes/alignment/capacity_key、initial_values及terminal_readers，不能只记录resolved V |
| accesses | ID/task/source occurrence/operand boundary、真实root/view、mode/owner、guard、generation、AccessRelation的relation/values/may_access/affine对应；原生view语义从source重读并与该记录核对 |
| effects | ID/task/source、kind、access_ids有序关联、state_ref、generation/guard、completion、pipe；内部无buffer状态与控制效果也保留 |
| edges / reuse | 每条方向、真实端点引用、distance、delay、kind与完整effect集合；Reuse完整last_reader/next_writer/buffer/distance_offset/delay，保留同buffer多条，不能dict覆盖；当前Reuse端点是access ID，沿access.task_id核对时序 |
| capacities / 策略 / coverage | storage key、buffers、usable/reserved/source/status；issue_after/manual_pipe_order、offset开关、stage域、fixed/auto II界；coverage全部expected/covered集合与refusals；complete派生值不作权威，不hash可变诊断detail替代关系 |

coverage集合本身不证明依赖正确。每次认证还需source_audit从实际IR独立重建本支持域reaching-def与依赖，复核problem；保存一个包含错误边的新hash不能让错误关系合法。

## 5. 特殊值：明确编码或具名拒绝

| 当前identity遍历的缺口值 | 最小可实现表示/拒绝规则 |
|---|---|
| live Module/Stmt/Expr/Var、循环Region引用、对象图环 | source的定义/occurrence表编码back-reference；局部is关系在消费时另核，不递归dataclass到循环；对不在source表中的foreign ref拒绝 |
| lambda、bound method、任意Python callable或opaque对象 | 不编码__name__/repr/地址；事实用其已调用后返回的typed值，注册op用显式注册身份；数据字段真携带callback则拒绝，无通用callback序列化；numpy scalar/array、torch Tensor或设备句柄也不能以repr入字节，首版只接受已转换为显式typed字面量的Constant，否则拒绝 |
| Target实例 | 注册类名+语义参数+上述实际facts查询返回值；**新实例同参数同facts应得相同target_identity**，但旧plan要求使用原target_ref，新实例须重新resolve绑定；行为不同但参数相同由facts变化检出 |
| AccessRelation/isl.map | 只用本build维度API：读取PARAM/IN/OUT/DIV维数，逐basic_map/输出读取plain_get_dim_aff(OUT,k)，以(dim_type,index,coefficient)位置记录全部系数、常数项及get_div的局部除法定义；有域约束时还需完整记录等式/不等式及其常数/系数，缺少可读取表示即具名拒绝。禁止字符串parse或任何参数重命名路径 |
| relation.values / affine | named=sorted(_parameter_names(relation))必须与values键集合完全对应，并与PARAM各位置读取到的名字一一一致；按named排序位置编码values[name]绑定的真实Expr规范字节，名字本身不入语义hash。名字/位置不一致或参数经exists转为局部维度或被消去（不再是PARAM）却仍提供values，返回带字段路径IDENTITY_UNSUPPORTED_VALUE，不猜对应。affine=None显式编码；非None同样按位置/系数编码并核其与relation一致 |
| ISL的多种等价表示 | 固定当前ISL build，以位置/系数元组编码basic_maps及局部div；同一表示内无序约束按规范系数字节排序。首版只保证同构同bytes，不承诺一般数学等价归一；同一对象两次编码不一致即拒绝，不能通过打印/重命名“修复” |
| Enum/DType/DimVar、set/map、type字符串 | Enum按已登记类型+value，DType按真实描述符字段，DimVar按声明符号+lo/hi与source维度绑定；容器按§2。无已登记encoder的class/type/layout拒绝，不能以__str__兜底 |

ISL payload只含维数、位置、精确系数/常数、约束/div与对齐的Expr绑定，不包含重命名文本；它仍不是依赖等价证明。系数使用精确整数或分子/分母表示（整数可num_si，大整数/有理数可用isl.val的数值字符串），不得float舍入；逐位API用法见[现有lowering:1022](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/passes/transforms/convert_hir_to_tir.py:1022>)，含get_domain_local_space().get_div(...).get_coefficient_val(IN,...)与fix_val。AccessRelation.values在[实际定义:58](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/visitor_registry/access_relation.py:58>)中为可变dict，冻结外层record也不会冻结它；:108生成的__tf_id名字必须从身份字节消除。

## 6. 固定锚、重算与失败边界

| 时点/消费者 | 重读对象与比较方式 |
|---|---|
| UnitTree构造 | 从真实lowered module/entry及target_ref生成source bytes与固定snapshot；明确snapshot.entry_ref is module.entry_function()；后续不因变化自动重建这个锚 |
| analyze / resolve | 先重读source+facts等于旧snapshot；独立source audit与赋值postcheck成功后生成problem bytes；coverage.problem_identity一致性单独验证；将problem_ref、固定problem bytes与最终plan bytes封存在同一ResolvedPlan |
| completion build / verify / storage / apply / native verify | 从plan.problem_ref重读source IR、target facts、完整problem；分别等于SourceSnapshot.canonical_bytes、plan.problem_canonical_bytes；从实际计划字段重算plan bytes等于plan.plan_canonical_bytes，并核H(bytes)等于相应固定identity。然后独立重验本次语义义务，不信上次VERIFIED |
| require(completion) | 若保留单参数，则completion持有由真实build/verify绑定的本地plan_ref/target_ref，require走同一重读流程与verify；目标与plan.problem_ref.target_ref必须is。没有本地绑定时拒绝，不靠receipt/hash发许可证 |
| install / export / bind | 在各自进程可访问的实际对象/descriptor边界重新检查对应source/plan/artifact/TL/ABI内容；本地安装仍须同一个已验证candidate对象，不能以内容hash相同替代is绑定 |

变化返回具名失败：SOURCE_SNAPSHOT_STALE、TARGET_FACTS_CHANGED、PROBLEM_CONTENT_CHANGED、PLAN_CONTENT_CHANGED、SOURCE_BINDING_MISMATCH或IDENTITY_UNSUPPORTED_VALUE。检测发生在首次读取该输入的消费边界，不给每层另加“已检查”缓存标志。正常无变化记录可成功重算，不无条件拒绝全部程序。
frozen dataclass只限制字段赋值，不冻结IR、dict、Target返回表或op实例；is只证明引用同一对象。正确比较对象是**当前真实对象重新规范化的bytes**与resolve/分析先前固定的bytes锚；problem.coverage.problem_identity只是待核字符串，不能作为锚的唯一来源。
即使调用方协调替换problem与自报hash，也必须同时匹配旧source快照、plan固定bytes以及真实source audit和整数postcheck；hash机制不负责抵抗任意篡改整个Python进程的代码执行，不发明额外registry/token框架。新source/target/problem需要新一轮分析与resolve，而非更新旧plan中的锚。

## 7. 可直接落测试的判定清单

| 测试构造（真实对象） | 身份/字节预期 | 必须发生的消费结果 |
|---|---|---|
| 相同输入两次编码；map插入序改变；合法局部alpha改名 | 同一source/problem/plan bytes；公开symbol或ABI角色不变 | 新分析可同内容身份成功；老plan的foreign对象仍不能冒充旧refs |
| 同名不同binder换operand；改局部Var.type、op常数、guard、For stop、UB view byte offset、output_count | source变，继而新problem/plan身份变 | 老plan在source重读处拒；不可用同名/同shape修复 |
| 删除/增加一条access或改mode/root/view | 旧source不变、当前problem bytes变 | completion/verify/require先拒problem stale；重新resolve仍由audit判断缺/多项，不因重hash恢复完整性 |
| 改relation.values绑定Expr、位置系数/常数/div、域约束/may_access；仅改参数名字但排序位置与Expr绑定均保持 | 前者problem变；后者系数/绑定字节相同。参数名排序与PARAM位置不一致、exists消除参数后残留values均无合法bytes | 老plan拒真实变化；不一致返回IDENTITY_UNSUPPORTED_VALUE含路径；正例覆盖PARAM=0/values={}、精确div和同对象重复编码，不进行字符串重命名实验 |
| 改effect completion/pipe/guard/access关联，删/反向edge或改distance/delay/effect集合，改reuse offset/端点、删除同buffer某末读者 | problem bytes变，source可不变 | 老plan拒；新计划需独立依赖/完成证明，不能只过集合coverage |
| 改ResourceUse.offset/duration、capacity/reserved、Fixed/Auto域或slot_bytes | problem变；新resolve计划身份也必须绑定新problem | completion/storage/apply/require拒旧plan；model仍SAT也不能重用旧认证 |
| 改stage/II/phase/start/issue_rank/origin/bounds/resolved V，保留原plan_identity字符串 | 真实plan bytes变 | PLAN_CONTENT_CHANGED；即使时序仍可行也重新resolve/认证 |
| 同一target实例返回不同UB容量/flag池/cost；或换实例有不同facts | target及source envelope变，problem/plan随之变 | 旧绑定拒TARGET_FACTS_CHANGED；未知容量保持显式未知，不回填 |
| 换同类同参数同facts的新target实例 | canonical target/source内容可相同；对象绑定不同 | 老plan消费拒SOURCE_BINDING_MISMATCH；用新实例完整分析/resolve可成功，不能把类名当完整配置 |
| 原地改IR/AccessRelation.values但保持对象is、声明hash不变；放入lambda/陌生layout | 前者真实bytes变化；后者没有合法bytes | 所有消费边界拒旧对象变化/不支持值；无repr兼容降级 |

当前未运行这些新测试；首个单region支持域可先验证正例与上述突变，不扩展为一般跨region/carry证明。事实身份与来源语义审计两条证据均齐全后，才有可消费的plan.problem_ref。
