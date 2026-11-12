# TileFoundry Ascend950PR 原生编译与单 kernel 工作包 v3

**用途：** 交给本地 Agent Team，基于当前 main 完成原生调度框架、attention_residual 与融合 MLA，并在真实环境逐步测试和验收。

**审计基线：** `f4d6fe11769974bb31396acf890acc710749de5c`。版本日期：2026-11-12。本包是完整新版实施规格及核心实现参考；不是已完成 NPU 实现，也没有直接修改 GitHub 仓库。

## 先读这三个文件

1. `specs/00_architecture_and_decisions.md`：整体结论、实际进度和架构决定。
2. `plan/PHASES.md`：8 阶段、34 任务及关键依赖。
3. `prompts/START_TEAM.md`：可直接复制的完整启动提示词。

合并阅读版是 `MASTER_SPEC.md`。它包含全部 8 章 spec、阶段计划、关键实现移植说明与验收边界。可执行 Python 代码、原 fixtures 和机器可读任务表在本目录；分发时请发送整个 ZIP。

## 核心改变

正常入口应真正消费作者策略和共享 solver 结果，把符号流水安装为实际 entry。所有原生 tasks/effects、SSA、versions、completion 和 physical storage 使用同一计划。backend 执行该计划并核对 TL 剩余 lowering；两个算子都经这条路线，不能以独立文本 twin 或直接 TL 旁路验收。

静态 4×64 乘法微例是工具链基础。G61 协议模型、G69 模拟计时和旧 CPU 测试数量不代表完整方案。工作包附本次实际复现的 solver、text checker 和 AR 数学/参数反例。

## 内容索引

| 目录/文件 | 内容 |
|---|---|
| specs/00–07 | 架构、native 作者接口、分析/符号流水、solver、completion/storage/backend、两个算子、验证/性能、接口迁入 |
| implementation/ | 4 个关键框架实现：固定 II Z3、符号 loop 构造、native binder 克隆与 entry 安装、编译链编排 |
| tests/ | 35 个框架 CPU 测试；不冒充 native/NPU 验收 |
| reference_core/ | 20 个独立数学/协议测试及 AR/MLA 核心算法模型 |
| fixtures/ | 18 份原 golden/test/输入/helper/许可证；hash 保持不变 |
| cases/ | 原 40 矩阵与数值合同、独立 supplemental MLA 诊断 |
| plan/ | 34 任务 DAG、8 阶段门、阶段说明 |
| prompts/ | Lead 启动词、角色任务词、恢复提示 |
| templates/ | 初始进度、搜索预算示例；时间预算由本机 P0 校准 |
| tools/ | 原 runner 连接器、包校验、可执行任务查询 |
| evidence/ | 源码审计、可复现反例、实际 CPU 结果、来源与测试边界 |

## 快速使用

从包根目录运行：

```sh
python tools/verify_package.py
python tools/next_tasks.py
python tools/run_contract.py --help
```

独立算法测试需要 Python 3.12、NumPy 与 Z3 绑定，版本见 requirements_cpu.txt。真实 TF/CANN 环境应复用本机已兼容组合，不用这个小依赖文件重建生产环境。

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s reference_core -p 'test_*.py' -v
```

原源码反例可在固定 f4 基线的只读工作树上复现：

```sh
python evidence/reproduce_schedule_counterexamples.py --repo /absolute/path/baseline --output /absolute/path/schedule-probes.json
python evidence/reproduce_twin_counterexample.py --repo /absolute/path/baseline --output /absolute/path/twin-probe.json
python evidence/reproduce_operator_counterexamples.py --repo /absolute/path/baseline --output /absolute/path/operator-probes.json
```

这些脚本仅诊断该基线，不对已修改的新分支假装给出相同结果。修复回归应迁到新仓库正常测试路径。

## 完成标准

原 20+20 例通过；每次算子调用单逻辑 kernel；真实计划对应、资源与同步通过；AR 三主族与 MLA canonical/列片/deferred 均真实实施并参与规定搜索；最终 dispatch 完整复跑 40 例；另一个 agent 独立复现。多 kernel 调度仍留后续。

本包 CPU 测试与用户已有 CANN 微例证据分开记录，详见 evidence/VALIDATION_STATUS.md。不要把文档里的目标函数名、示意配置或待实现 hook 当成当前 main 已有能力。
