# 符号物化、whole-task 克隆与安装规范

本规范承接[已定字段](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md:68>)与[冻结编译事务](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:34>)；源码基线 f341e47529453bdbc8dd81f2c108d36628fb4b8b。下列为实现合同，不表示当前 apply/verify/install 已实现。只读设计，仅写本文。

## 1. 输入、结果与执行顺序

apply_symbolic_pipeline(entry,plan,completion,storage)消费同一source snapshot、schedule identity、plan_iteration和Target事实下的完整UnitTree/bindings、SAT ResolvedPlan、CompletionPlan、StoragePlan，返回**私有候选 Module + SemanticSourceMap**；entry是原Module实际PrimFunction，不是metadata中的替代入口。
原Module与entry及其原生子对象始终不修改；新函数、source_map、分析附着和metadata容器都属于候选。共享未改的不可变输入允许，原对象原地改写不允许；不能先更改source再以更改后的source作预期。
顺序固定：物理allocation准备 → 按region内向外符号转换 → verify_actual_program → 在同一plan附NativeVerdict → emit_supported_tl/read_actual_tl与实际refinement认证 → install_entry。NativeVerdict只证明native对应，安装须等完整ConformanceVerdict。
不再求解或选择V，phase/II只决定已求出的发射顺序，不生成等待估算周期的指令；同步机制、字节映射和尾部义务全部来自输入计划。

## 2. 符号迭代与准确边界

每个region取a=lower、b=stop、h=step，h是已证正整常量；a/b保留原生Expr。计算N=max(0,floor((b−a+h−1)/h))；o=minᵤ(stageᵤ)，shiftᵤ=stageᵤ−o，H=maxᵤ(shiftᵤ)，须与RegionPlan.origin/max_shift相同。空region不求min，沿已定空region语义保持原控制。
构造一个原生For：p∈[0,N+H)，step=1。按issue_rank递增，每个完整单元只建一个body：kᵤ=p−shiftᵤ，iᵤ=a+kᵤ·h，Dᵤ=(0≤kᵤ<N)。原loop IV按身份替换为iᵤ；实际计算有效域为Dᵤ∧原guard[iᵤ]。
Dᵤ必须先形成外层If。仅在其then域内求原guard及其依赖；不能依赖布尔and短路语义去避免对无效迭代的load。原guard内嵌控制保持原位置；只有已证纯且总定义的索引表达式可用于D外的Let。

| 情形 | 实际符号形态与义务 |
|---|---|
| N=0，H=0 | p∈[0,0)；无单元body执行，尾FREE循环为空；本region无Set/Wait/drain |
| N=0，H>0 | p∈[0,H)可保留H个空步；所有Dᵤ恒false。只计算纯k/i，不读operand/原guard，不执行通知；不将H误作真实迭代数 |
| N=2、V=4 | 每单元k=0、1各执行一次；writer没有k≥V旧FREE等待；每代末读者均发FREE，尾g∈[0,2)消费两代，而非遍历4个槽 |
| N=2、stages=(2,7) | o=2、shifts=(0,5)、H=5，p∈[0,7)；首单元在p=0/1，后单元在p=5/6；p=2/3/4为空隙，不能减掉gap或压成两步 |
| a=3、b=10、h=2 | N=4，逻辑i=3/5/7/9；slot按数据generation决定，不按i%V或p%V；仅在已证有效数据代次域中形成实际view/地址，空步负k不生成访问 |

改变N只改变bounds/运行次数，UnitTree单元、solver变量、completion模板与候选TIR节点数量不按N增长。正向小N用实际trace核验，大N用实际IR结构统计；允许纯常量折叠，但它不得形成按N复制body的另一条路径。

## 3. 按绑定身份克隆whole-task

环境分为外部捕获、局部binder与stream视图三部分，key均为定义对象身份；不同调用occurrence、分支和循环作用域使用独立环境。每个whole VF/DMA/Cube/composite保留完整Sequential/For/If/Let/Evaluate/ExecutionScope/MeshScope结构和注册内部合同。替换范围是选定入口中的region；未声明转换的其他函数、callee签名与调用目标保持原对象/语义，不能借候选重写它们。

| 节点/值 | 唯一允许的处理 |
|---|---|
| 参数/捕获 | 未替换的外部Var保留原身份；需要映射者来自RegionBindingSummary，不克隆出未绑定的同名Var |
| Sequential中的Let | 先在旧环境重写value，再建fresh Var并重写type/annotation，随后仅对后续sibling扩展env；自引用不凭名称修复 |
| For / MeshScope | bounds/mesh表达式在外env重写；fresh IV/binding仅进入body；退出恢复外env；ExecutionScope保留kind与执行域 |
| If | cond在当前合法域求值；两臂各用独立env副本，分支内binder不泄漏；原guard不能被移动到更大执行域 |
| Call / Evaluate | 保留实际op、全部operand顺序、attribute参数、精度、mask与effects；仅重写已声明的binding/generation/address，不按access表重建算式 |
| type / layout / view | 同env重写所有shape维度、layout参数、view offset/pitch与原Expr；旧type/annotation中的IV不可残留，不能默认f32/linear |
| 跨单元/跨代值 | 按已分析的root、value stream及generation映射到StoragePlan视图；纯索引可重物化，含load/状态读取的值必须有真实定义或显式stream |
| 局部临时值 | 只在原合法VF/mesh/guard内freshen；寄存器不跨VF逃逸；不能将所有allocation或GM参数都改成ring |
| 嵌套region | 先完成内层自身转换与exit，外层只把它作为具有完整边界效果的注册composite；不将内层动态循环flatten为外层单指令任务 |

stream allocation先迁到能支配全部使用者且仍在原合法执行/激活域内的存储作用域，按正式StoragePlan建一次真实allocation；task内对应旧定义仅由该已计划视图替代。迁移不是执行effectful初始化：初始化、clear或读取依旧按原代次与guard运行，不能因为hoist allocation而提前执行。
纯Let外提须同时满足无memory/state/control effect、操作无trap、在目标域全部变量已绑定且表达式总定义。除这些纯索引外，load、配置、随机、状态读取、实际算术及effectful Let都留在D与原guard的合法计算域内；guard所需效果依其源先后原样执行。
本树[BindingSubstitutionCloner](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/visitor.py:380>)可复用identity-keyed表达式替换，但它不是TIR fresh-binder转换器：默认未映射Var会被克隆，且memo只按源Expr身份，不区分env。whole-task处理须保留外部Var身份，并每个独立env/occurrence创建新cloner或用(env,Expr)联合memo。
[StmtExprMutator](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/visitor.py:510>)结合dataclasses.replace可重建语句，但默认遍历不重写绑定字段、type/layout或实际op属性；需要显式binder/type/mesh处理。不能照搬测试中跳过type/annotation的局部突变器作为生产cloner。
标量p/k/i/guard必须使用TF纯值运算及已注册typeinfer/evaluator/printer/backend支持；带dst的TIR Binary是效果指令，不能伪作纯标量表达式。遇到无合同节点/动态step/无法表示的跨域值，在转换前的完整问题构造阶段具名拒绝。

## 4. completion与storage注入的唯一位置

注入遵循CompletionSite.anchor与真实pipe完成合同；source计算guard和通知的balanced envelope分离。D内依次为pre-wait、原guard内whole计算、post-set；具体多事件顺序由同一completion模板给出，不能仅按字母或event ID排序。

| 注入位置 | generation、控制域与物理操作 |
|---|---|
| 单元BEFORE_TASK | 在D内先等所有READY/STATE；覆盖写入前若k≥V等相应FREE(k−V)。等待的是通道记录的数据代次，不是当前task ordinal或当前p |
| 单元原计算 | 原guard及effectful Let保留在有效域；每条access用其计划generation Gᵦ(k)，真实槽为Gᵦ(k) mod Vᵦ |
| 单元AFTER_TASK | 在D通知域内发布本代READY/FREE/STATE；事件在对应pipe的写可见/末读完成点生效，不能把“已issue”当完成。原guard为false时按已证balanced模板执行空通知，读guard⇒写guard仍须成立 |
| FREE尾部 | p循环之后、region live-out消费之前，每个末读通道保留原生For g∈[max(0,N−V),N)；在该通道原接收pipe/执行域Wait，generation由g及stream偏移映射，不枚举所有N |
| region exit | 尾FREE消费后，按completion.exit_site_ids排布未覆盖的异步store/状态drain；此region N>0才产生其循环遗留工作，独立边界效果保持自身原激活域 |
| storage视图 | 先取root身份对应PlacementEntry，地址=base_offset+(Gᵦ mod Vᵦ)·version_stride+已重写的原view局部offset；layout/alignment/执行域沿用正式结果，不重新分配 |

Vᵦ只取ScheduledBuffer.resolved_versions，CompletionChannel.versions与slot_count必须等于它。generation完整表达代次，slot只是有限物理索引；两代slot相等不表示同一值/同一token。event ID由物理namespace/base加该通道generation%V形成，READY/FREE不重新拆池。
N=0不执行的是循环任务和其遗留同步；原region外部初值、激活控制和live-out保持原语义，零次循环yield仍取初值。carried S的初值S[0]在原region入口执行一次，迭代k可读S[k]、写S[k+1]；写slot=(k+1)%V、读slot=k%V，初值和最后yield按各自value generation定位。尾g是归一化producer ordinal，真正FREE token可能对应g+delta，不能将g直接当数据generation。
一个flag token若是消费式，多个独立消费者使用各自channel；合并通知必须来自既定完成合同。scratch按独立scratch_id/copies分配一次，不能随task克隆重复建，也不能把copies替代buffer V。

## 5. source_map：可验证的对应，不是替代程序

SemanticSourceMap的顶层source/plan/candidate identity与typed字段一致；SourceMapEntry保留source_ref真实节点、unit_id、candidate_refs真实节点、binder_pairs、substitutions及COMPUTE/CONTROL/ALLOCATION/SYNC角色。相同DAG节点的不同调用occurrence分别沿已定source定位表识别，不能用id(source)单一字典吞掉多重调用。

| 来源内容 | map必须覆盖的落点 | verifier如何消费 |
|---|---|---|
| 每条源statement、op与operand occurrence | 对应克隆节点、fresh绑定与有限替换；原计算/control均有来源 | 从冻结source独立遍历预期，按subst重写后比较candidate的实际op/attrs/全部operand，检查每个有限模板恰对应一次 |
| 每条AccessFact与Effect | 以已有source_ref+unit_id+INPUT/RESULT/operand_index或状态定位，关联对应COMPUTE节点及其原生view/代次；id沿plan原表反查 | 重读该candidate实际访问/状态，逐项验证root、地址域、mode、guard与generation；不能用map声称“已覆盖”替代读取 |
| p/k/i、valid、原guard与nested control | CONTROL角色记录生成控制节点与source binding、substitution | 证明每个task的(k,i)有效实例与原iteration一一对应，非单纯比较节点数 |
| hoist allocation、views、state初始化/出口 | ALLOCATION或CONTROL角色记录原allocation/边界源与StoragePlan派生关系 | 比较真实allocation次数/域/字节/stride、初值与最后yield，不从同shape查root |
| wait/set/tail/drain | SYNC角色记录真实节点，unit_id沿模板所属单元，site/channel经completion的真实引用回查 | 比较端点、控制域、物理namespace与generation/slot，所有新增效果都须有可解释来源 |

source_map记录“计划允许的重写”，verify独立读取“实际发生的重写”。candidate中的每个计算和效果必须归入源或明确的allocation/sync/control派生；不能丢掉原statement，也不能把新增算术隐去为辅助节点。多重性按模板与符号域证明，不按N展开事件列表。

## 6. verify_actual_program：真实对象上的证明范围

调用verify_actual_program(original,rewritten,plan,completion,storage)，original是冻结源对应原Module，rewritten是刚产出的私有候选Module；map通过同schedule identity的plan.semantic_source_map显式传递，不新增隐式metadata入口。返回NativeVerdict，记录整个Module与entry的不同identity及实际快照/读取事实。

| 判定维度 | 必须基于实际source/candidate读取的内容 |
|---|---|
| 来源完整与调度实现 | 全部源单元/statement/access/effect覆盖；p/k/i、bounds/step、stage shift、issue_rank与计划相符；每个有效iteration的任务恰一次，N0没有效果 |
| 绑定与算式 | binder定义/使用域、fresh局部Var、全部operand顺序、op属性、常数/精度/type/layout；vmul(a,b)改成vmul(b,b)即不对应，hash相同或名称相同无证明力 |
| guard与效果 | 外层D先于原guard依赖，effectful Let在原有效域、原分支/嵌套循环保留；提升allocation没有提升初始化或实际读取 |
| 状态与完成 | 初值/循环carried/yield对应；READY/FREE的全部末读者、writer wait、N<V尾部、跨核端点和退出drain；slot与generation分别比对 |
| 物理实现 | 实际root/执行域、allocation/视图字节、V/stride/offset与completion/storage同迭代；scratch只计一次，旧offset未被保留为权威 |
| 正规TIR合法性 | 使用verify_prim_function(candidate_entry,module_fns=candidate_module)检查真实类型/mesh/ExecutionScope/SSA作用域；其通过不替代语义覆盖证明 |

独立CPU验证读取安装后或私有候选的实际TIR，以两组不同非平凡输入对照原语义；当前通用HIR evaluator不能直接执行PrimFunction，必须使用覆盖本支持域的TIR解释器。已有测试oracle只支持有限指令，能力不足应扩展执行语义，不能改成返回golden或把metadata当执行体。
语义核对、CPU数值、TL conformance分别报告；NativeVerdict.VERIFIED不宣称设备数值通过。任何失败都舍弃候选，保持原Module.functions、entry、原生对象内容与可调用语义；记录拒绝原因不发布任何候选工件。

## 7. 安装边界与发布重核

**本地对象绑定补充：** 对现有NativeVerdict增加original_module_ref:Module、candidate_module_ref:Module、candidate_entry_ref:PrimFunction三个编译进程内必需字段；真实verify设置这些引用，字段compare=False/repr=False，ConformanceVerdict.native_verdict沿现有plan→artifact→reader链保留同一结果。它们不进canonical hash/跨进程descriptor，不创建新schema或API；此前字段规范的快照/hash继续绑定内容。仅有hash无法证明“同一个对象”。
reader在真实candidate来源绑定上完成TL认证；编译进程将返回的actual TL读取事实与其保留的同一NativeVerdict合并成完整ConformanceVerdict后才安装。跨worker传回的是读取事实，不自行构造native引用；不以反序列化的新PrimFunction冒充已验证对象。
install_entry沿现有conformance参数消费该结果；module必须is verdict.original_module_ref，rewritten_entry必须is verdict.candidate_entry_ref，candidate_module.entry_function()必须is该entry；当前内容快照必须仍与两份verdict相符。语义相等、同名、同hash的新函数也不是此次已验证对象。
只把Module.functions中**is原entry**的唯一位置替换为该candidate entry，保留其余函数、entry名、目标与拓扑；新Module.entry_function()必须is rewritten_entry。原Module保持不变，metadata拷贝后才附验证结果；不能把functions缩成仅入口而丢callee。
公开导出与bind前，从实际安装entry、完整ABI（含output_count/参数类型顺序）、plan/completion/storage、实际TL source/parsed/refinement、双键PassConfig、target/toolchain内容重核绑定；同时原Module内容仍等于其认证前快照。cache hit也取得同一认证语义，不据缓存状态跳过边界。

## 8. 最小机器验收集

真实正常入口执行：N=0/1/V−1/V/V+1/17、非零lower、step>1、stage gap>N；百万N比较实际IR规模。两组不同输入在真实TIR执行得到正确非平凡结果。多nested region、同名不同binder、同shape不同root、effectful Let、carried初值/末yield均覆盖。
对实际候选分别改变operand/属性、遗漏或复制计算、移动guard/Let、改slot/generation、漏writer wait/尾wait/末读者、换执行域或layout offset，native或TL对应必须失效；只改注释与metadata不能恢复资格。另给同名同内容fresh入口，install必须因对象不属于已验证candidate而拒绝。
所有成功证据关联实际source→candidate→TL→installed entry对象/内容，不把helper模型、自报VERIFIED、测试收集或零测试当本段完成。本次只制定规范，没有执行上述验收。
