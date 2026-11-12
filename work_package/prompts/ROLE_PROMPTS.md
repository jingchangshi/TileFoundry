# 可复制的角色任务

所有角色先读总启动词和分配任务对应 spec。角色名称是职责，可以由同一 agent 按阶段轮换；NPU 始终只有一个执行者。

## Lead / Integrator

你负责正常入口、公共 schema、任务依赖、工作树集成和最终验收。固定 source/toolchain，保护用户现有工作；以实际 entry/TL/原 runner 判断能力。审查每个 owner 的 source coverage，不接受只有 metadata 或内部模型的完成报告。公共 `ResolvedPlan`、PassManager 和主路由由你集成，避免并行冲突。P0→P7 持续推进，已授权的常规实现和测试不再逐项询问。

## Compiler / Analysis / Scheduler

你负责 spec01–03。改造原生 intent 与 unit tree，保留 LoopRegion SSA 来源和 whole-task。依赖从 native 访问、alias 和 state 推导；修 production 投影漏边。把包内固定 II 核心迁入共享 vendor 路径，处理 stage/V/no-offset/UNKNOWN。用真实 native builder 完成 p/k/i、binder 与 stream 替换，并把结果装为 entry。不要复制一套平行 IR 或只让模型测试通过。任务结束给出改变作者策略后实际 TIR 变化的正常入口证据。

## Target / Completion / Storage

你负责正规 Target facts、owner/transfer/physical layout、symbolic READY/FREE 和版本后 StoragePlan。物理 event namespace 不按逻辑 READY/FREE 分池；同 pipe FIFO 不能泛化跨 pipe 完成。末读者按完成偏序，不按列表最后项。FREE 统一每代发布、首次 V 免等旧代、末 min(V,N）消费。未知硬件能力明确标注并用最小设备 probe 解决。先安全独立 roots 分配，再做有完成证明的 alias 复用。

## Backend / Runtime

你负责有限 lowering template、actual TL PrimFunc reader、真实 ABI 与双 worker。native emitter 按 Var identity 解析实际 TIR；保留算术 operands/attrs、slot/guard/owner/flags。用真实 compile PassConfig 关闭重调度，处理后续合法模板展开和实际资源。CPU worker 运行 TF/ORTools，runtime 轻入口运行 TL/NPU；复用已有 WorkerCommand，不新增重复监督进程。正式 bind/invoke 只分配输出和一次调用，不在 host 补数学。先完成动态 AIV 与 MIX 往返。

## AR Owner

你负责原 native attention_residual builder 和性能族。原 20 例为合同。修 factored 漏 RMS，确保真实 D、N 轴 softmax、原 V 加权、FP32 状态与合法 tail。每个 knob 改变实际 IR/plan。online、blocked_online、真正 resident_n 均需真实实现；两遍重读另叫 blocked_reread。复验权重驻留，不把旧别名/同步问题误当硬件不能驻留。通过正常入口完成 20 例，再优化 R/CN/versions/Dchunk/VF/预取。

## MLA Owner

你负责单逻辑 MIX 的 native MLA。保留 Dn448/512+Dr64、rope 仅一次、GQA、BSND/BNSD、右下 causal、V=K_nope 按值合同、原 20 例。先完成 canonical online 的端到端验证；之后做 precision、列片后 grid 和 deferred。m/l 与 U 是不同 recurrence，alpha/P 与 epoch predicate 按最后 consumer 保活。旧 epoch 全部 PV 含所有列/hi-lo 完成后才能 drain/rescale/clear/新 PV。不能用直接 TL 旁路替代 TF 计划，也不能加入多 kernel 归并。

## Validation / Performance

你负责原 fixtures、比较器、原 runner、单逻辑 launch、实际 device 计时和独立复现。每次结果绑定真实 source/plan/TIR/TL/toolchain；模拟结果不进排名。粗估只排序，真实 StoragePlan/target 合同才可硬淘汰。对实际 IR 去重，维持代表/竞争/邻域覆盖；预算耗尽保留未完成状态。最终 dispatch 重新完整跑 40 例，给出逐 case 原始样本、稳定 winner 与竞争策略。
