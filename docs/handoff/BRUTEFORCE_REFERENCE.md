# BRUTEFORCE-REFERENCE — 固定 II 调度的独立穷举参照（task-17）

- 作者：explorer-g1。**独立参照**：模型与约束完全按 task-17 规格自行推导，未参考任何 solver 侧代码（本树 solver 本身是 fail-closed stub，solver/__init__.py:26-27 raise，无约束实现可参考）。
- 产出：docs/handoff/probes/bruteforce_schedule.py（**160 行**，自包含，零第三方依赖）+ 本报告。基线 HEAD=f341e47。

## 怎么跑

```bash
/home/s00653124/workspace/envs/tfnpu-v2/bin/python docs/handoff/probes/bruteforce_schedule.py
# 15/15 项打印 PASS，最后 "SUMMARY: ALL PASS"，exit 0（实测 2026-11-12）
```

## 纯数据视图（字段名是本脚本的，等 solver 落盘后加适配层）

`stage_max`；`tasks`：name/pipe/latency/resources(额外资源单位)/stage(作者钉定或 None)，列表序=源序；`edges`：u→v, d(迭代距离), L(生产延迟)；`buffers`：name/space/slot_bytes/writer/readers/read_latency/versions=("fixed",n)|("auto",(lo,hi))/shift(δ)；`resources`：每资源每 II 周期并发席位；`capacity`：每存储域字节。

## 实现的约束（与规格逐条对应）

1. `t = II·stage + phase`，`0 ≤ phase < II`（域构造保证）。
2. 数据边：`t_v + d·II ≥ t_u + L`。
3. 版本复用：`t_writer + (V+δ)·II ≥ max_r(t_reader_r + read_latency_r)`（多 reader 取最晚完成）。
4. 资源环形区间：任务占 `[phase, phase+latency)` 圆弧（按单位数加权），**跨 II 回绕拆两段**做事件扫描，任意点覆盖 ≤ 容量；latency ≥ II 视为覆盖全周。
5. 同 phase 同资源：issue_order 按声明（源）序确定性地排序输出（SAT 赋值里带 issue_order）。
6. 容量：`Σ V·slot_bytes ≤ capacity[space]`。
7. 作者 stage 钉定：非 None 的 stage 不进枚举域。

枚举域：每 task (stage∈[0,stage_max])×(phase∈[0,II))；每 buffer (V 域)×(δ 域)；全笛卡尔积逐点判定——只适用于小域（这正是参照的用途）。UNSAT 返回尝试点数与命中的违约束类别集合（edge/version/ring/capacity），便于 validator 定位。`min_ii(p, ii_max)` 从 II=1 升序找第一个 SAT，并返回"所有更小 II 显式 UNSAT"的证明表。

## 自测（期望值全部手算写死；15 项全 PASS）

| 测试 | 手算期望 | 覆盖 |
|---|---|---|
| T1 | 单任务 II=2 → SAT @(0,0) | 基线 SAT |
| T2 | A→B L=1：II=1 UNSAT；optimal II=2 且证明 {1:UNSAT} | 数据边 + 最优 II |
| T3a/T3b | fixed V=1 放得下→SAT 且 V 恒=1；reader 钉 stage1 后 fixed V=1 **必须 UNSAT**，auto(1,2) 则 SAT V=2 | fixed V 不得被放大 |
| T4 | 两条 3 拍任务 1 席位：II=4、5 UNSAT（3+3>周期），optimal II=6 | **跨 II 环形冲突** |
| T5 | 两块 10B slot：smem=15 UNSAT（20>15）、30 SAT | 字节容量 |
| T6 | A 钉 stage1 → B 落 (1,1)；B 钉 stage0 → UNSAT | 作者 stage 钉定 |
| T7 | 2 席位同 phase SAT，issue_order 随声明序翻转 | 同 phase 源序 |

## 与 solver 实际签名的对齐状态

- 本树无 solver 签名可对齐：tilefoundry_ascend.solver 只有 `solve_region(problem, deadline=None)` 的 raise-stub 与空 `Assignment` dataclass（status/starts/stages/phases/versions/issue_order/optimality 全默认），没有任何 problem schema。
- 本参照的 `Assignment` 形状刻意与之同构：SAT 赋值含 per-task (stage,phase)（对应 starts/stages/phases）、per-buffer (V,δ)（对应 versions）、issue_order；对齐只需一层纯函数适配（problem 视图 ↔ solver schema），不含约束逻辑。
- 规格差异点留给适配层：本参照的 L 在边上而 solver 可能用 task latency；δ 域本参照默认 fixed 0（规格的 Fixed/Auto 语义在 INTERFACE_FREEZE 中属 BufferSchedule，域枚举已支持）。
- 独立性声明：约束公式与环形判定由本任务从 task-17 文字规格独立推导；若 solver 实现与本参照在同一小域上给出不同 SAT/UNSAT，以两者差异为准做仲裁（这正是交叉验证的目的）。
