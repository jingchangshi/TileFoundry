# 06 真实环境验证、性能搜索与验收

## 6.1 五种结果分别判断

| 结果 | 证明内容 | 不能替代 |
|---|---|---|
| CPU 数学/协议测试 | 公式、有限状态机、反例行为 | native 接线、设备数值 |
| native 入口集成 | 原策略与 solver 解改变实际 entry/TIR | 后端实际执行与精度 |
| plan 对应 | 真实 TL 实现支持域内与 TIR 策略一致 | 原 golden 数值 |
| 原 runner 数值 | 固定输入、seed、golden、比较器通过 | 单调用、容量与性能 |
| NPU 性能/trace | 指定设备环境中真实执行与计时 | 全局最优或其他环境性能 |

每个里程碑报告这些字段，不用一个总 `VERIFIED` 遮住未执行项。历史证据不改写；当前验收只指向本轮具体 commit 和产物。

## 6.2 环境与进程边界

P0 记录：TF 源码 commit 及 dirty 状态、Python、Torch、torch_npu、TileLang 源码/build、CANN 真实路径、驱动/设备查询、比较器 module 及版本、实际编译 PassConfig。沿用可工作的 CANN9.2 环境起步；不为迎合规格更换驱动。

CPU compiler worker 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，运行 TF/ORTools，产出正式 HIR/plan/TIR 与显式 TL。runtime worker 以轻入口独立 exec，加载 torch_npu/TileLang，不导入 TF/ORTools 分析链。已有 worker timeout/receipt/cache 设施继续使用。

所有设备工作由唯一 executor 持现有锁。并行 agent 可编辑、分析、跑 CPU 测试；NPU 候选按队列执行。不要让同一把锁在 supervisor/worker 重复获取。疑似挂起记录本任务状态并停止，不自动 reset 设备或终止其他任务。

## 6.3 原始数值合同

包内 `fixtures/` 保存 18 个原始文件，`fixtures/source_manifest.json` 记录 hash。`cases/case_matrix.json` 含 40 个精确 key/seed/shape/dtype/attrs。每个算子 20 例，不接受重复 key、漏 case 或改阈值。

原 test 文件提供 `main(operator_api)`，不是简单执行脚本就自动测试。包内可运行入口：

```sh
python tools/run_contract.py --help
python tools/run_contract.py --case attention_residual --operator-file /absolute/path/operator.py --callable attention_residual --device npu:0 --report /absolute/path/ar-result.json --log /absolute/path/ar-run.log
python tools/run_contract.py --case mla --operator-file /absolute/path/operator.py --callable mla --device npu:0 --report /absolute/path/mla-result.json --log /absolute/path/mla-run.log
```

这些是包内已存在的工具参数。正式运行应使用本机兼容解释器、已设置的 CANN 环境与设备锁。runner 默认关闭 oracle cache；它会校验原文件身份和完整标签。`--preflight-only` 只检查依赖/环境，不能记为数值通过。

原 helper 依赖外部 `npu_compare`。P0 应找到并记录实际模块；缺失时提交确切依赖请求和已经准备好的运行命令。不能编写宽松替代比较器，也不能用新的 allclose 阈值顶替原合同。输入生成、MLA 的 `get_input` 规范化和 golden 调用仍由原 harness 完成。

supplemental MLA 诊断覆盖 Nkv>1、显式 scale、layout 和 tail，位于 `cases/supplemental_mla.json`。它们用于补缺，不算入原 40 例。

## 6.4 编译器的关键反例

以下每类至少有一个真正作用于正常入口/真实产物的测试：

- fixedV2 被改成 V1；manual stage2/3 被归零；partial stage 缺省 0；no-offset 漏限制。
- 原生访问/配置效果未进入问题；read→write 方向错误；read/read 误作 RAW/REUSE。
- 改变 loopstop 却只改变 metadata；符号 body 未安装；N 增大导致 IR 线性增长。
- 同名不同 binder 和同 shape 不同 buffer 被串线；tail 或 effectful Let 被移到 guard 外。
- finalize 后将 `vmul(va,vb)` 改成 `vmul(vb,vb)`；少一个或多一个 arith 操作。
- 修改真实 slot、guard、flag endpoint、versionstride、physical layout；只改注释不能修复失败。
- 提前 FREE、遗漏末 reader、跨 pipe 把 issue 当完成；N0/短 loop/尾部 token 失配。
- deferred 中旧 epoch 的某列/hi-lo PV 迟到；factor/predicate 提前覆盖；不同 sid 走不同通知分支。
- compile worker 意外加载 torch_npu，或 runtime worker 意外导入 ORTools。
- search 传入模拟计时、重复实际 IR、未兑现 knob 或旧 run_id 结果。

不要为了新的支持域保留“功能尚未实现就应报错”的旧测试。更新其语义边界，并增加应成功的正常入口例子。测试应定位真正风险，不用数量证明完成。

## 6.5 纵向设备样例

先做动态 AIV 完整链：两个不同输入、非平凡算术、合法 tail、V1/V2/V3、stage0 与多 stage，覆盖 N=0/1/2/V−1/V/V+1/17 及大 N，lower 非零、step>1。比较实际 GM 输出并检查编译规模；不要继续限定 4×64 作为生产域。

MIX 样例应完成整个往返：AIC QK 或小 matmul→L0C/FIX→AIV UB 运算→UB→L1→AIC matmul→AIV 输出。它验证 layout、dtype、sid 分片、真实 completion 和一次逻辑调用。只编译一个 Cube 语句或只测 AIV 阶段不算通过。

静态微例可保留用于 cold/cache、环境和 import 隔离检查。它们不再承担完整 AR/MLA 的阶段信用。

## 6.6 性能搜索的输入和输出

复用 `search/generator.py/filter.py/harness.py/process.py`。将 driver 接到实际 native builder 和正常编译入口，再经独立 runtime worker 完成 numeric/launch/profile。设备排名应拒绝 `SIMULATED/MODEL_ONLY/INCOMPLETE` 计时。

候选真实 identity 由规范化策略、实际 native 语义/plan/布局/精度/target/backend 组成。只改名字、seed、flag 编号或未使用 knob 不算新候选。先比较实际产物，避免当前 AR 多个配置生成同一个 TL 仍计入搜索。

过滤分三类：

| 类型 | 作用 |
|---|---|
| 数学/ABI/明确 target 不合法 | 可以硬拒绝，记录具体依据 |
| 当前模板未支持 | 标 IMPLEMENTATION_MISSING，形成实施任务，不能当硬件不可行 |
| 粗略资源/性能预测 | 用于排序；最终由真实 StoragePlan/编译资源复核 |

例如 AR 纯 AIV 候选没有 L1 分配，不能按虚构 L1 ring 淘汰；Qrope 不应重复计费；同一 Cube 串行不等于 MTE 预取与 operand 生命周期不重叠。

## 6.7 搜索覆盖与预算

保留 v2 的初始覆盖目标：每算子至少 8 个代表，每代表至少 8 个实质不同、数值正确的设备候选；每个原 case 至少 3 个可行竞争候选；代表 winner 有局部邻域。初始每 case 40 候选，分为 12 seed、20 neighbor、8 advanced；再设总编译/运行/墙钟预算。

这不是填表配额。先实现全部应比较的核心族，再以 device 结果展开邻域；duplicate/fail/未实现不计数。预算耗尽时保留最快合法候选和剩余任务，状态是性能阶段未完成，不以“预算没了”豁免一个算法族。

若目标容量或原比较器实验证明具体族在某 shape 不可行，应记录准确域和证据，不能推广到全部 shape。原 40 例仍需有合法 dispatch。若某个 case 可行的不同候选实际不足 3 个，应提交源级可行域与测量记录供审阅，不能通过改名凑数。

代表应覆盖 AR 的大 D/非对齐/小 token/大 token/大 N/FP16 和 MLA 的 decode/MTP/prefill、两 layout、两 dtype、Dn448/512、causal。P0 从包内真实矩阵固定代表 key 并说明覆盖，不从错误 note 推断实际 shape。

## 6.8 计时与选择

编译、加载、输入生成、输出分配、oracle 和 host readback 在 kernel 计时区间之外。预热后使用已验证的设备 event 或 profiler；如果后端不支持某种计时器，应换可验证的设备路径，不能用 CPU 模拟时间。

在相同输入和同一环境下交错测试候选，记录原始样本、median 及离散程度。差异接近噪声时重测，选择稳定且资源较低的候选，不宣称小于噪声的提升。kernel 时间与 operator 端到端时间分别报告。

baseline 是本轮第一个经过原 runner 验证的单 kernel 实现。附件 optimization 可以作为另一个历史对照，但不是最优目标。性能报告逐 case 给出 baseline、winner、两个实际竞争候选、资源、编译时间、策略和瓶颈；可同时汇总 geomean，但不掩盖单 case 回退。

## 6.9 最终门

1. 最终 dispatch 从所有合法 shape/dtype/layout 选择恰好一个已验证 artifact，不执行 host 数学修补。
2. 清理本任务缓存或显式禁用后，重新编译并运行原 20+20，不沿用开发阶段挑选日志。
3. 再完整运行原 40 例作 cache-hit 复验；核对同一 source/plan/ABI/flags 的绑定，不能复用错误 shape 的 kernel。
4. 对最终 specialization 检查实际 TL/CCE、launch、target 资源和计划对应；所有数值例都来自原 runner。
5. 第二个 agent 在同一已锁环境按文档独立复现，给出完整结果或明确失败。
6. 交付当前 commit、命令、完整产物与缺口。未完成项不写 PASS，不将缺设备伪装不可实现。

只需一份当前 run manifest 连接这些文件。无需为每层再建一套独立“权威证据计划”。
