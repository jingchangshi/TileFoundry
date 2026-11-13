# TileFoundry Ascend Target 插件

`tilefoundry-ascend` 为 TileFoundry 提供 `ascend` 目标后端支持，涵盖 Ascend 950PR / DaVinci 950 架构与设备建模、Ascend TIR 指令集以及基于 Z3 的 Modulo 流水线求解与验证 Pass。

## 安装方式

```sh
python -m pip install -e . -e targets/ascend
```

## 包结构说明

- `tilefoundry_ascend.ascend.target`：文档驱动的 `DaVinci950`（`dav_3510`）架构模型、`Ascend950PR` 设备模型、`AscendTarget` 组合目标及硬件 Facts 投影（`TopologyFacts`、`MemoryHierarchyFacts`、`AscendSpaceFacts`、`ThroughputFacts`、`PerformanceServiceFacts`）；支持模式感知的三级 UB 容量（`SimtVF` `216 KiB` / 纯 `SimdVF` 默认 `248 KiB` / 开启 `PURE_SIMDVF_UBUF_FLAGS` 时 `256 KiB`）。
- `tilefoundry_ascend.ascend.ir`：`SimdVF` 与 `SimtVF` 执行模式声明，以及 50 条内置 `typeinfer`、ISL `access_relation`、CPU 模拟求值和语句校验的 `T.ascend_*` TIR 指令（覆盖 DMA/Reg、4 类扩展 `Reshard` 数据重分布、Cube MMAD、`dual_copy`、统一 HIR 二元/一元/归约/条件/裁剪指令、谓词指令与标量-向量混合指令及 `vmula` FMA）。
- `tilefoundry_ascend.ascend.visitor_registry`：面向 `MatMul`、`Cast`、统一数学 HIR（`HirBinary`/`HirUnary`/`HirReduce`/`HirWhere`/`HirClamp`）及 `Reshard`（16 个目标作用域候选）的 HIR->TIR 候选指令绑定。
- `tilefoundry_ascend.ascend.schedule`：Z3 Modulo 流水线约束投影与求解器（`pipeline_solver.py`），支持循环携带依赖抽取、模式感知（`216/248/256 KiB`）与双 AIV UB 切分、DMA 字节与 Cube FLOPs 周期延迟估算。
- `tilefoundry_ascend.ascend.passes`：面向 `AscendTarget` 的 `ValidateAscendPipelines`（`BEFORE_LOWER`）与 `ResolveAscendPipelines`（`AFTER_LOWER`）模块 Pass。

## 相关文档

- [Ascend 950PR 总体架构与 MLA 端到端流程指南](../../docs/reports/ascend950pr_architecture_and_e2e.md)
- [Ascend 950PR 适配与 SimdVF 性能优化报告](../../docs/reports/ascend950pr_adaptation.md)
