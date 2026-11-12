# 原生编译链 typed schemas：字段与不变量

规范基线：f341e47529453bdbc8dd81f2c108d36628fb4b8b；本文件细化[冻结合同](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:6>)，不是第二套 API。下列为实现目标，不声称当前均存在。仅新增/演进现有编译记录，无另一套算子 IR；本文未修改源码或调用设备。

## 公共类型、可选性与来源

表内每个字段默认**必需 R**；仅标 ? 的字段类型含 None，None 的唯一含义在该行给出。tuple 可空仅表示已知空集合；未知不以空集合表示。int 是 plain int、不含 bool；正整数>0，字节/距离/模型时间非负。所有正式记录及其嵌套集合冻结；Expr/原生引用仍需与不可变语义快照绑定，frozen dataclass 本身不冻结 IR。
Ref 是真实对象引用；同进程按 is/定义身份匹配，禁止用名字、shape、打印path或zip循环顺序回查。ID 是分析按词法定义及调用 occurrence 分配的非空稳定标识；跨进程序列化保存ID表与正式语义内容，不保存id()数值或pickle。Hash是canonical语义内容的SHA256，不含对象repr/内存地址/诊断metadata；它绑定内容，不证明等价。
OwnerKey=(program_group:Expr,core_kind:AIC/AIV/DEVICE,sid:Expr?)；AIV的sid必填（单路为常量0），AIC/DEVICE的sid必须None，不以物理core编号替代逻辑program。StorageKey=(space:str,owner:OwnerKey)；ResourceKey=(owner:OwnerKey,pipe:str,instance:int)；pipe必须由Target声明。谓词都是已类型化纯bool Expr，恒真也用原生true Expr；索引使用纯整数Expr。
VersionDomain 是唯一带标签联合：Fixed(value:int>0) 或 Auto(minimum:int>0,maximum:int≥minimum)，不得同时持两种状态。model_version:Hash标识约束/时间刻度/目标cost profile；模型整数tick与实测ns分开。SourceSnapshot={module_ref:Module,entry_ref:Function/PrimFunction,canonical_bytes:bytes,identity:Hash}，bytes由正式语义遍历冻结；所有后续记录绑定该identity。

| 来源简称 | 当前已有记录/能力与限制 | 喂养的消费者 |
|---|---|---|
| B | LoopRegion.params/args/yield_values/captures/carries与lowering memo、loop_nodes；现有[binding summary](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/passes/transforms/convert_hir_to_tir.py:2231>)为dict，尚需typed化 | RegionBindingSummary、UnitTree、克隆/source map |
| A | [Access/SnapshotEntry/BufferAlias](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/analysis/access.py:70>)、[IterationScope.access/provenance](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/analysis/iteration_scope.py:212>)、[AccessRelation](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/visitor_registry/access_relation.py:47>)；**无AccessMetadata类型**。AccessRelation.values显式绑定每个符号参数 | root/view/alias、精确访问域与coverage、依赖推导 |
| E | [ParamDef.MemoryEffect](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/core/param_def.py:44>)与TIR access注册、实际operand及目标ISA状态/完成合同；effect=None是未声明，不是纯操作；STATE/async generation需新增推导 | 单元effects、问题边、completion与source verifier |
| M/T | [MemoryMetadata](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/analysis/metadata.py:186>)的operands/footprint/buffer_bytes + 原生TensorType/type/layout + Target space/event/layout facts | 逻辑shape/footprint与物理需求；traffic汇总不能重建精确访问边；旧offsets不是最终版本placement |
| C | [ComputeCostMetadata](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/analysis/metadata.py:119>)只有flops/other_ops/precision，结合目标吞吐/指令资源与显式cost_override推导模型tick | latency/reservations/delay，不能把flops直接当latency |
| L/Q | [LivenessMetadata](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/analysis/liveness.py:49>)的uses/intervals/regions + [AllocationSummaryMetadata](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/analysis/allocation.py:158>)的values/in_place/address_decisions及root/alias解析 | 来源寿命/跨域值/逻辑allocation；结构事件时间不是异步read completion，最终LiveWindow必须由completion推出 |

SourceLocator 是带标签联合 Native(node_ref:Module/Expr/Stmt,occurrence_id:ID) 或 Text(source_identity:Hash,begin_byte:int≥0,end_byte:int≥begin_byte)，用于Refusal定位，不承载语义。
WireExpr={kind:CONST/REF/CALL,dtype:str,value:bool/int/float?,binder_id:ID?,op_id:str?,args:tuple[WireExpr],attributes:tuple[(str,bool/int/float/str)]}；CONST只含value，REF只含binder_id，CALL只含op_id/args/attributes，其余字段None/()；不支持opaque表达式。ProgramFacts/runtime descriptor中的Expr与OwnerKey递归编码为WireExpr及WireOwnerKey(group:WireExpr,core:AIC/AIV/DEVICE,sid:WireExpr?)，运行侧不导入TF类型。WireStorageKey=(space:str,owner:WireOwnerKey)，WireMechanism按Mechanism同字段递归替换其key/Expr；所有枚举值与native记录一一对应。

## 1. RegionBindingSummary 与 UnitTree

生产：normalize_intent保存HIR绑定，ConvertHIRToTIR补实际引用，build_native_unit_tree形成whole-task树；来源B/A/E/T/L。消费：analyze_native_problem、apply、source verifier。

| 记录 | 字段名：类型 | 不变量/字段来源 |
|---|---|---|
| RegionBindingSummary | source_region:LoopRegion；induction_var:Var；start,stop,step:Expr | B；induction_var is source_region.induction_var；bounds来自lowering映射，stop不是未经换算的extent；step为正整常量 |
| 同上 | carried_params:tuple[Var]；carried_initials:tuple[Expr]；yield_values:tuple[Expr] | B；三者等长m=len(source.yield_values)，严格对应params[:m]/args[:m]/yield_values |
| 同上 | captures:tuple[(Var,Expr)]；actual_loop:For；actual_induction_var:Var | B；captures严格等于params[m:]/args[m:]；actual IV来自该source loop的lowering identity map |
| 同上 | binding_map:tuple[(Expr,Expr)]；actual_defs:tuple[Var]；actual_allocations:tuple[Var]；actual_bounds:tuple[Expr,Expr,Expr] | B/Q；map覆盖captures/carries/init/yields与IV，每source occurrence唯一目标；defs/allocs含合法嵌套定义；bounds等于actual_loop.start/stop/step；不按loop数量zip |
| UnitTree | source_snapshot:SourceSnapshot；entry_ref:PrimFunction；roots:tuple[Region]；units:tuple[SourceUnit] | checked TIR+B；roots只含函数级Region，嵌套region仅沿direct_children可达；每unit在一个direct_children出现一次，units仅为同一对象的有序索引 |
| Region | region_id:ID；node_ref:PrimFunction/For；parent_id:ID?；direct_children:tuple[SourceUnit/Region] | B+真实TIR结构；仅函数根parent=None；direct_children严格为whole-task语义层级，不是按指令flatten |
| 同上 | induction_var_ref:Var?；lower_expr,stop_expr,step_expr:Expr；binding_summary:RegionBindingSummary?；activation_guard:Expr | loop用B且全部非None；函数根IV/binding=None、bounds=(0,1,1)形成一次域；无伪造LoopRegion；activation来自外层真实控制 |
| SourceUnit | unit_id:ID；node_ref:Stmt；region_ref:Region；source_order:int；whole_body:Stmt；kind:VF/DMA/CUBE/COMPOSITE | 真实checked TIR+B；whole_body是该node完整计算，source_order在兄弟列表唯一且保序；composite具有注册的内部资源/effect合同 |
| 同上 | execution_owner:OwnerKey；resources:tuple[ResourceUse]；original_guard:Expr；captured_values:tuple[Expr]；carried_values:tuple[(Var,Expr,Expr)]；yields:tuple[Expr] | execution/mesh+B/L；resources显式非空；guard是原计算谓词，不混valid(k)；carried为param/init/yield，captures按定义绑定 |
| 同上 | effects:tuple[Effect]；intent:TaskIntent? | E+PipelineSpec；intent=None仅作者未声明；所有内部效果归属本whole unit，纯单元也保留语义与资源覆盖 |

无正整常量step、无法保持binder/嵌套绑定、未注册composite在来源分析边界产生带source定位的拒绝；不能填step1、捕获空表或用名字完成映射。

## 2. NormalizedProblem：完整原生问题与数值投影

生产：analyze_native_problem(units,effects,alias,target)，来源A/E/M/T/C/L/Q；消费：solver adapter、source postcheck、resolve_schedule、completion。以下辅助记录均属问题本身，不是source-only sidecar。

| 记录 | 字段名：类型 | 不变量/来源 |
|---|---|---|
| NormalizedProblem | identity:Hash；source_snapshot:SourceSnapshot；target_identity:Hash；model_version:Hash；unit_tree:UnitTree；regions:tuple[RegionProblem]；coverage:CoverageVerdict | 同一source/target/cost快照；每个RegionProblem完整覆盖其直接调度单元；空region可保留但不提交core；identity按源快照/typed原生内容/数值域计算，排除coverage字段防递归；coverage.problem_identity绑定此值 |
| RegionProblem | identity:Hash；region_ref:Region；tasks:tuple[Task]；buffers:tuple[LogicalBuffer]；accesses:tuple[AccessFact]；effects:tuple[Effect] | 完整有限集；所有ID引用闭合，refs回到UnitTree/A/E/Q原对象；跨region以注册composite/边界单元参与外层问题，内外效果关联而非遗漏；无因UNKNOWN被省略的效果 |
| 同上 | resources:tuple[ResourceFact]；owners:tuple[OwnerKey]；pipes:tuple[str]；edges:tuple[Dependence]；reuse:tuple[Reuse]；capacities:tuple[Capacity] | T+C+A；resources含所有owner/core/pipe域；edges/reuse按关系推导而非同root笛卡尔积 |
| 同上 | issue_after,manual_pipe_order:tuple[(ID,ID)]；enable_offset:bool；max_stage:int≥0；fixed_ii:int>0?；ii_min,ii_max:int>0 | PipelineSpec+同域源序；关系(a,b)含义a先发射b；fixed=None仅自动II，有限ii_min≤ii_max；固定值时min=max=fixed；stage域0..max_stage与V域独立 |
| Task | task_id:ID；unit_ref:SourceUnit；uses:tuple[ResourceUse]；latency:int≥0；source_order:int；authored_stage:int≥0? | C/intent；uses=(resource:ResourceKey,duration:int>0,offset:int≥0)，offset+duration≤latency；None保留“作者未标”，manual sibling缺省0在core规范化，不冒充作者显式0 |
| ResourceFact | key:ResourceKey；capacity:int=1；source:str；status:REFERENCE_IMPLEMENTATION/DEVICE_VALIDATED | T；当前core只接受capacity1；多lane资源需明确映射为独立可证明资源，否则分析域外 |
| LogicalBuffer | buffer_id:ID；root_ref:SSAValue/Var；allocation_ref:Expr；owner:OwnerKey；space:str；dtype:DType；shape:tuple[Expr]；layout:Layout；domain:VersionDomain；generation_delta:int | A/Q/T/BufferSchedule；root/alloc identity唯一；GM ABI非ring，未显式声明时domain=Auto(1,1)，不得伪造作者Fixed；stream默认仅未单独声明时Auto(1,max_buffer_versions) |
| 同上 | slot_bytes:int>0；alignment_bytes:int>0；capacity_key:StorageKey?；initial_values:tuple[Expr]；terminal_readers:tuple[ID] | M/T+lowering requirements；bytes已padding/aligned；None capacity_key仅不进入有限片上容量约束的GM边界；carried初值显式，末读者为完成偏序的极大集合；单一delta仅支持所有访问同一固定代次偏移，不满足则拆独立逻辑stream或分析域外 |
| AccessFact | access_id:ID；task_id:ID；source_ref:Expr/Stmt；boundary:INPUT/RESULT；operand_index:int≥0；root_ref:SSAValue/Var；view_ref:Expr；relation:AccessRelation；mode:READ/WRITE/READ_WRITE；owner:OwnerKey；guard,generation:Expr | A+E；boundary明确input_index还是result/output_index，operand_index为该边界位置；operand与view真实；relation参数全绑定；READ_WRITE同时进入读/写分析；may_access必须保守覆盖，不能略去 |
| Effect | effect_id:ID；task_id:ID；source_ref:Expr/Stmt；kind:MEMORY/STATE/CONTROL；access_ids:tuple[ID]；state_ref:Expr?；generation:Expr；guard:Expr；completion:ISSUE/WRITE_VISIBLE/READ_COMPLETE/STATE_COMPLETE；pipe:str | E+目标完成合同；MEMORY引用access且state=None；STATE引用真实state且access可空；CONTROL可二者空但有实际操作；未知效果不是空元组 |
| Dependence | producer,consumer:ID；distance:int≥0；delay:int≥0；kind:RAW/WAR/WAW/STATE；effect_ids:tuple[ID] | A/E；t_consumer+d·II≥t_producer+delay；WAR是旧读→覆盖写；read/read无数据边；跨region关系必须进入显式region边界效果/composite，不伪填distance1 |
| Reuse | last_reader,next_writer,buffer:ID；delay:int≥0；distance_offset:int | A/E/generation_delta；t_next+(V+offset)·II≥t_last+delay且V+offset≥1；每个极大末读者一项，不替换逻辑RAW |
| Capacity | storage_key:StorageKey；buffers:tuple[ID]；usable_bytes:int≥0；reserved_bytes:int≥0；source:str；status:REFERENCE_IMPLEMENTATION/DEVICE_VALIDATED | T；Σ(V·slot_bytes)+reserved≤usable；reserved包含未列buffers的固定scratch/保留allocation，不能再把同一字节计入buffers；相同底层allocation不重复计费；UNKNOWN无可构造usable整数 |
| CoverageVerdict | source_identity,problem_identity:Hash；expected_tasks,expected_accesses,expected_effects:frozenset[ID]；covered_tasks,covered_accesses,covered_effects:frozenset[ID]；refusals:tuple[Refusal] | 期待集合独立遍历冻结源，covered由实际problem refs反向得到；Refusal=(code:str,source_ref:SourceLocator,detail:str)，完整=三组相等且refusals为空；不得手填PASS |

core的Task/Buffer/Edge/Reuse/Capacity/Problem仅是上述记录的数值投影：name=对应ID、resource=ResourceKey的无歧义编码、owner_space=StorageKey编码；source_ref保留原生引用供TF侧，数值核不解释它。core不能成为第二份原生计划。未表达域在analyze_native_problem返回coverage拒绝，保留原源；完整problem才供正常链solve。
Assignment字段冻结为status:SAT/UNSAT/UNKNOWN、ii:int>0?、starts/stages/phases:不可变映射[ID,int]、versions:不可变映射[ID,int>0]、issue_order:tuple[ID]、optimality:Optimality、problem_identity/model_version:Hash、detail:str；SAT时ii非None且映射覆盖全部task/buffer、issue_order是task全排列，非SAT时ii=None且赋值集全空、optimality两bool均False。deadline是调用参数float绝对monotonic时间/显式None，不放进problem语义身份，不被任何region重置。
Optimality={ii_checks:tuple[(int>0,SAT/UNSAT/UNKNOWN)],ii_optimal_within_bounds:bool,memory_optimal_for_ii:bool}；最小II只在所有较小候选已证UNSAT时为真，首个SAT不代表内存最优；已有SAT与其未证最优性可同时存在。core Result可用assignment:CoreAssignment?封装相同值，TF adapter只导出上述唯一Assignment视图。固定II core/adapter实现与公开solve_region有限搜索是不同交付范围；本节规定后者的目标字段，不表示其deadline/三态/来源覆盖已接通正常链。

## 3. ResolvedPlan 与符号来源映射

生产resolve_schedule，仅接受SAT、独立整数postcheck和完整coverage；来源完整problem+赋值；消费completion/storage/apply/emitter/verifier。schema只有这一份，所有版本字段由model原样导出。

| 记录 | 字段名：类型 | 不变量 |
|---|---|---|
| ResolvedPlan | source_snapshot:SourceSnapshot；source_identity,target_identity,problem_identity,identity,model_version:Hash；plan_iteration:int≥0；normalized_intent:tuple[PipelineSpec]；regions:tuple[RegionPlan] | identity为source/target/problem/model/iteration与已决schedule内容的hash；不含后续completion/storage/source_map/native_verdict，避免递归hash；source_identity等于snapshot.identity |
| 同上 | task_to_native_source:tuple[(ID,SourceUnit)]；logical_buffers:tuple[LogicalBuffer]；access_and_state_effects:tuple[Effect]；feasibility:SAT；optimality:tuple[(ID,Optimality)] | 引用同一problem原记录，不能由JSON重新构造业务；feasibility非SAT时根本不构造ResolvedPlan |
| 同上 | completion_templates:CompletionPlan?；physical_storage_plan:StoragePlan?；semantic_source_map:SemanticSourceMap?；native_verdict:NativeVerdict? | None只表示本编译阶段尚未产生派生结果，不能export；不可变演进以replace附加同plan identity/iteration的结果，不重新resolve或改变V |
| RegionPlan | region_id:ID；native_region_ref:PrimFunction/For；lower_expr,stop_expr,step_expr:Expr；ii:int>0；origin,max_shift:int≥0；tasks:tuple[ScheduledTask]；buffers:tuple[ScheduledBuffer] | bounds与Region同语义；origin=min(resolved_stage)、shift=stage−origin、max_shift=max(shift)；无task的空region origin/max_shift=0且ii=1，仅用于保持控制来源 |
| ScheduledTask | task_id:ID；unit_ref:SourceUnit；authored_stage:int≥0?；resolved_stage,phase,start,issue_rank:int≥0 | start=ii·resolved_stage+phase、phase<ii；issue_rank为region全排列顺序且符合hard issue边；manual绝对stage保持、未标缺省0；no-offset全0 |
| ScheduledBuffer | buffer_id:ID；root_ref:SSAValue/Var；domain:VersionDomain；resolved_versions:int>0 | Fixed时等于value，Auto时位于[min,max]；域引用原声明/分析明确默认，GM边界V=1；该V由completion/storage/slot/TL共同消费，不再计算另一V |
| SemanticSourceMap | source_identity,plan_identity,candidate_identity:Hash；entries:tuple[SourceMapEntry] | apply返回私有candidate Module与本map；candidate_identity重算实际candidate全部函数/入口ABI、target；map不是APPLIED状态 |
| SourceMapEntry | source_ref:Expr/Stmt；unit_id:ID；candidate_refs:tuple[Expr/Stmt]；binder_pairs:tuple[(Var,Var)]；substitutions:tuple[(Expr,Expr)]；role:COMPUTE/CONTROL/ALLOCATION/SYNC | candidate_refs是真实clone/生成节点；whole-task含原全部计算；辅助sync/hoist有对应来源/派生合同；subst含p/k/i、generation/slot及type/layout所需变换；statement多重性由有限模板对应验证，不按N列实例 |

## 4. CompletionPlan：有限模板与物理事件

只用一个CompletionPlan类型表达冻结合同的SymbolicCompletionPlan，生产build_symbolic_completion(plan,target,effects)显式消费E；消费placement/apply/actual verifier/TL。L中的文本last-use不是FREE时刻。

| 记录 | 字段名：类型 | 不变量 |
|---|---|---|
| CompletionPlan | identity,plan_identity,effects_identity,target_identity:Hash；plan_iteration:int≥0；sites:tuple[CompletionSite]；channels:tuple[CompletionChannel]；exit_site_ids:tuple[ID] | 同plan迭代；identity覆盖全部sites/channels/物理机制，不是数量；模板数量不随N增长；无可手填的verified字段 |
| CompletionSite | site_id,region_id:ID；task_ref:SourceUnit?；anchor:BEFORE_TASK/AFTER_TASK/REGION_EXIT；task_ordinal,generation,guard:Expr；owner:OwnerKey；pipe:str | REGION_EXIT才允许task_ref=None；其task_ordinal是该exit尾循环归一代次g或region终点N，generation独立显式；其余site必须绑定真实unit |
| CompletionChannel | channel_id:ID；buffer_ref:SSAValue/Var?；state_ref:Expr?；reason:READY/FREE/STATE/EXIT；producer:ID；consumers:tuple[ID]；versions:int>0；physical_mechanism:Mechanism | producer/consumers引用sites，均非空；READY/FREE有buffer且state=None，versions必须等于其ScheduledBuffer.resolved_versions；STATE有state、buffer可None且须引用该state在plan中的版本结果（非stream为1）；EXIT可二者None且versions=1，依据实际exit效果 |
| Mechanism | kind:LOCAL_EVENT/CROSS_EVENT/FIFO/PIPE_BARRIER/DRAIN；namespaces:tuple[EventNamespaceKey]；base_id:int≥0?；proof_source:str | EVENT有namespaces/base且为每bank同时占用[base,base+V)；FIFO/BARRIER/DRAIN base=None，namespaces=()，仅精确目标完成合同可构造 |
| EventNamespaceKey | owner:OwnerKey；core:AIC/AIV/DEVICE；pipe_pair:tuple[str,str]?；bank:int | 同核core等于owner.core_kind且有pipe_pair；跨核core=DEVICE、pair=None、owner=OwnerKey(关联program_group,DEVICE,None)，接收SID不再拆池；bank为非负实际bank编号，每bank单独一个key，bank mask分解后同时预留；READY/FREE不进入key |

有限模板规定：每代Set/Wait控制域平衡；计算保留原guard且读guard蕴含写guard；writer k≥V等FREE(k−V)，每代极大末读者发布FREE，尾部g∈[max(0,N−V),N)在原接收pipe/执行域wait；N0无通知。依赖上的STATE/EXIT来自真实效果，不凭空补channel。一个硬件flag若是消费式通知，多个pipe/域消费者分别建立channel，不能对同一token重复Wait；同pipe合并须有完成偏序依据。未知bank/capability/跨域guard关系在completion构造处具名拒绝。

## 5. StoragePlan：字节与完成寿命

生产place_resolved_versions(plan,completion,target,lowering)，来源M/T/Q+lowering requirements与completion；消费apply/TL/refinement/verifier。演进[现有PhysicalLayout/Requirement](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/model.py:263>)及placement记录，不平行定义第二套物理答案。

| 记录 | 字段名：类型 | 不变量 |
|---|---|---|
| PhysicalLayout | layout_kind:ND/NZ/FRACTAL/PACKED/SCRATCH/RESERVED；logical_shape,physical_shape:tuple[Expr]；dtype:DType；item_bytes,pitch_bytes,alignment_bytes,required_bytes:int>0；breakdown:Breakdown；source:str；status:REFERENCE_IMPLEMENTATION/DEVICE_VALIDATED；refinement_id:str? | dtype取真实type；required_bytes是单版本完整对齐需求；breakdown=(payload,padding,scratch,staging,reserved):非负字节，和为required_bytes；无额外refinement时id=None；布局明确，L1不猜linear |
| Requirement | root:SSAValue/Var；owner:OwnerKey；space:str；versions:int>0；layout:PhysicalLayout；version_stride_bytes,slot_bytes:int>0；extra_scratch:tuple[ScratchRequirement]；completion_edge_ids:tuple[ID]；invalidation_key,target_profile_digest:Hash | versions来自ScheduledBuffer；slot_bytes=layout.required_bytes；stride≥slot_bytes且满足alignment；主allocation字节=V·stride；key覆盖plan/effects/completion/shape/layout/target/requirements |
| ScratchRequirement | scratch_id:ID；storage_key:StorageKey；layout:PhysicalLayout；copies:int>0；live_window:LiveWindow；source_unit_id:ID | 每个实际scratch allocation恰一个ID；copies表达真实共享或每版本拷贝数，分配copies·layout.required_bytes一次；不能同时在主layout.breakdown.scratch与extra_scratch记同一字节 |
| LiveWindow | begin_site:ID；end_sites:tuple[ID]；generation,guard:Expr；happens_before:tuple[(ID,ID)] | begin为实际首写，ends覆盖全部极大异步末读/退出完成；同generation有效域；来自completion而非issue序号；跨root复用须有HB证明 |
| PlacementEntry | root:SSAValue/Var；storage_key:StorageKey；base_offset_bytes:int≥0；version_stride_bytes:int>0；slot_count:int>0；allocation_bytes:int>0；layout:PhysicalLayout；live_window:LiveWindow；requirement_identity:Hash | base按alignment对齐，slot_count=V、allocation_bytes=V·stride；slot地址base+(generation mod V)·stride；不同root范围首版不重叠，有证明alias才可重叠 |
| StoragePlan | identity,plan_identity,completion_identity,effects_identity,target_identity,lowering_identity:Hash；plan_iteration:int≥0；requirements:tuple[Requirement]；entries:tuple[PlacementEntry]；scratch_entries:tuple[ScratchPlacementEntry]；capacities:tuple[Capacity]；peak_bytes:tuple[(StorageKey,int≥0)] | ID覆盖真实全部布局/字节/寿命；scratch entries与scratch requirements逐ID对应、copies与真实allocation一致；已知容量每域peak≤usable；不带自报verified；None容量不得以footprint回填 |

ScratchPlacementEntry={scratch_id:ID,allocation_ref:Var,storage_key:StorageKey,base_offset_bytes:int≥0,copy_stride_bytes:int>0,copies:int>0,allocation_bytes:int>0,layout:PhysicalLayout,live_window:LiveWindow,requirement_identity:Hash}；bytes=copies·stride，stride≥layout.required_bytes且对齐；copies不是plan中的buffer V，不混用slot公式。

## 6. NativeArtifact、ABI 与读取事实的 verdict

生产emit_supported_tl从实际private candidate发射，真实parser/refinement读取，ABI由实际entry推导；消费read_actual_tl→install/export→bind_invoke_once。现有[ParameterSpec/OrderedABI](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/abi.py:75>)和[NativeArtifact](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/codegen/__init__.py:17>)原位演进；runtime记录不import TF分析类型，Expr在已决specialization中转换为已绑定整数，不支持的动态ABI在编译边界拒绝。

| 记录 | 字段名：类型 | 不变量/实际读取来源 |
|---|---|---|
| ParameterSpec | name:str；index:int≥0；role:tensor/scalar；storage:str；dtype:str；shape:tuple[int≥0]；strides:tuple[int≥0]；layout:str；alias_group:ID? | 实际params.type/layout推导；role实际枚举沿用现有小写tensor/scalar；scalar shape/strides=()、storage=scalar、dtype为精确scalar类型；tensor storage=gmem、显式元素strides与layout；alias=None只在已证独立，授权别名共享group |
| OrderedABI | schema:str；entry:str；parameters:tuple[ParameterSpec]；output_count:int>0；allowed_aliases:tuple[(int,int)] | 原schema唯一演进，parameters按callable顺序index=位置；最后output_count项为tensor outputs；tensors/scalars可以交错，不重排为tensors后scalars；derived input_count=len(params)−output_count包含scalars，tensor_input_count另数；digest从全部字段派生 |
| NativeArtifact | source:str；parsed_prim_func:实际TL/TVM PrimFunc；abi:OrderedABI；source_identity,plan_identity,tir_identity,tl_identity,completion_identity,storage_identity,target_identity,toolchain_identity:Hash | source为实际完整TL bytes的UTF8文本；tl_identity=sha256(source UTF8)，tir_identity等于native_verdict.entry_identity，是private候选实际entry语义含ABI/output_count/target全配置；parser必须从该source产生parsed，不由caller替换 |
| 同上 | entry_symbol:str；pass_configs:tuple[(str,bool)]；native_verdict:NativeVerdict；expected_facts:ProgramFacts；refinement_manifest:RefinementManifest | symbol等于ABI.entry；配置精确为两项tl.enable_auto_schedule=False、tl.disable_shared_memory_reuse=True；tuple顺序固定按此顺序，runtime真实compile消费同一值；expected从TIR/plan及native_verdict.observed_facts生成，不从TL注释读取 |
| ProgramFacts | identity:Hash；abi:OrderedABI；units:tuple[TemplateFact]；allocations:tuple[AllocationFact]；events:tuple[EventFact] | expected由实际candidate及source map派生，actual独立读取PrimFunc；TemplateFact具体字段见下一行，均为canonical typed结构，不是字符串注释 |
| TemplateFact | unit_id:ID；template_id:str；operands:tuple[WireExpr]；operand_roles:tuple[INPUT/OUTPUT/INOUT]；binders:tuple[(ID,str,tuple[int])]；attributes:tuple[(str,bool/int/float/str)]；loops:tuple[(ID,WireExpr,WireExpr,WireExpr)]；guards:tuple[WireExpr]；owner:WireOwnerKey；slot,generation:WireExpr | operand_roles等长对齐；binders含定义ID/dtype/shape；loops为IV/lower/stop/step，按实际词法次序；模板实例不随N枚举 |
| AllocationFact / EventFact | AllocationFact=(allocation_id:ID,storage_key:WireStorageKey,shape:tuple[int],dtype:str,base_bytes:int≥0,stride_bytes:int>0,size_bytes:int≥0,alias_group:ID?)；EventFact=(site_id:ID,mechanism:WireMechanism,generation/slot/guard:WireExpr,order:int) | 实际buffer/view字节与事件端点/代次/控制域逐一读取；cross-process为typed canonical表达式编码，不能传TF对象；具体binder标签按词法定义身份生成 |
| RefinementManifest | identity,toolchain_identity,pass_configs_identity,input_ir_identity,output_ir_identity:Hash；steps:tuple[RefinementStep]；final_resources:tuple[AllocationFact]；final_events:tuple[EventFact] | Step=(registered_rule_id:str,before_identity/after_identity:Hash,actual_ir_snapshot:bytes,resource_delta:tuple[AllocationFact],event_mapping:tuple[(ID,ID)])；actual snapshot来自真实pass观察，最终资源/事件覆盖实际IR或CCE；未观察≠空集合 |
| NativeVerdict | status:VERIFIED/REFUSED；source_identity,plan_identity,candidate_identity,entry_identity,completion_identity,storage_identity:Hash；source_map_identity:Hash；source_snapshot_bytes,candidate_snapshot_bytes:bytes；observed_facts:ProgramFacts?；refusals:tuple[Refusal] | verify_actual_program独立读原source与candidate，证明statement覆盖/多重性、operand/attr/type/layout/guard/binder/代次替换；observed=None仅REFUSED且实际IR不可读取，并有对应refusal；candidate_identity绑定整个Module，entry_identity单独绑定入口且和artifact.tir_identity同义；它只认证native对应，不单独授予install资格 |
| ConformanceVerdict | status:VERIFIED/REFUSED；artifact_identity,entry_identity,plan_identity,tl_identity,source_identity,completion_identity,storage_identity,target_identity,toolchain_identity:Hash；parsed_prim_func:实际PrimFunc?；native_verdict:NativeVerdict；expected_facts:ProgramFacts；observed_facts:ProgramFacts?；refinement_manifest:RefinementManifest?；compile_pass_configs:tuple[(str,bool)]；refusals:tuple[Refusal] | ?仅REFUSED且该真实输入尚未取得，有对应refusal；artifact.parsed_prim_func与reader参数必须同一个真实parser结果，传入refinement manifest与artifact绑定同一内容身份；reader实际比对expected/observed及登记refinement；status=VERIFIED当且仅当native证明成立、ABI/语义/完成/布局/资源/两配置全部对应、refusals为空；artifact_identity为此artifact全部语义字段hash，不含verdict自身 |

认证数据流固定：verify_actual_program产出NativeVerdict；事务编排用不可变replace将结果附到同schedule identity的ResolvedPlan.native_verdict，emit_supported_tl通过已有plan参数读取它，并连同实际candidate/source快照组装NativeArtifact.native_verdict，read_actual_tl(artifact,parsed_prim_func,refinement_manifest)直接读取它，逐一关联candidate/source/plan身份与expected_facts。它不读取隐藏metadata，也不增加第四个reader参数；ConformanceVerdict.native_verdict必须是artifact中同一结果。原入口与候选的活引用仅留编译进程，install以实际对象内容重核snapshot绑定后替换；runtime只接canonical快照/事实。
VERIFIED须携带实际读到的事实与内容绑定，不能凭typed外壳或verified=True；Hash相等也不构成证明。install消费完整ConformanceVerdict且原子替换Module.functions；export/bind再从当前entry/artifact重核对应内容，原source失败不变。预编译binary未观察可作为独立能力事实记录，不伪造binary hash，也不影响已观察的parsed/refinement范围表述。
每个表的生产/消费段也约束其辅助记录：ResourceUse由C/T形成供solver，Dependence/Reuse/Coverage由A/E形成供solver/postcheck，SourceMap由apply形成供verifier，Mechanism由T及completion形成供apply/TL，Requirement/LiveWindow由placement形成供发射，ProgramFacts/RefinementStep由真实IR reader形成供认证。
域外统一返回Refusal(code,source_ref,detail)；当前尚未实现的服务保留具名未实现。类型构造只表示字段合法，生产资格来自上述真实分析/读取；未知relation/完成能力/容量、未登记TL模板或refinement不能构造完整已认证结果。
