# BF16-PROBE（task-45）— 真实 bf16 作者程序能否降低 + parse（G2/S4 前置）

explorer-g1，只读 + parse-only（未 compile、未 import torch_npu、未开 /dev、未 git 写）。基线 HEAD=f341e47；工作树并行更新中（emit.py mtime 03:25、tl_reader.py 03:26、vector_model.py 03:22），本报告的每个行号都按**当前工作树**重跑核实。探针：/tmp/tf_fac/{bf16load,bf16e2e,bf16full,fp16load}.py + /tmp/tf_fac/{run_bf16,run_bf16full2}.py。

## 0. 结论

**"真实 bf16 程序可降低"= 可行；"可降低+可 parse 进支持域"= 目前不可行**，且卡点**不是 dtype 映射**（并行提交已补），而是：

1. **carrier 存储类别不被发射面承认**（当前唯一实测拒绝点）：`supported_domain → (False, "TL_BUFFER_SCOPE_MISMATCH", "carr is declared as a UB tile but lives in carrier")`（emit.py:313-317：scope=="shared" 时要求 storage=="smem"，carrier 直接拒）。
2. **vcvt / pset16 无 TL_SURFACE**：emit.py 的 TL_SURFACE 仍只有 6 条（copy/vld/pset/vadd/vsts/gm_to_ub/ub_to_gm），bf16 e2e 必经的 vcvt 两个方向与 pset16 都在下一步 TL_NO_SURFACE（emit.py:452）。
3. **16-bit lane 数属需硬件证据**：新 vector_model.py 把规则钉为部署工具链自己的 `_vreg_lanes = 2048 // element_bits`（引 tilelang/ascend/language/simd.py:85-118），故 64 元素 bf16 tile → **bfloat16x128 / extent 128**；该 128 是否硅上真打包**UNSOURCED pending device evidence**，reader 对 16-bit 只给 **INCOMPLETE、永不 VERIFIED**（vector_model 模块注释 + tl_reader.py:678-686）。

## 1. 真实 bf16 端到端 lowered 实测（作者语法，非手改文本）

bf16full：GM bf16 in → gm_to_ub → vld(UNPK_B16)→carrier → vcvt forward（pset16 mask）→ f32 vadd → vcvt reverse（pset16）→ vsts(pack="PK_B32") → ub_to_gm bf16 out。

```text
checked_program OK / analyze OK / FINALIZE OK
lowered params: [('x','bf16','GMEM'), ('out','bf16','GMEM')]
supported_domain: (False, 'TL_BUFFER_SCOPE_MISMATCH', 'carr is declared as a UB tile but lives in carrier')
torch_npu: False
```

指令面（lead ①）在作者链路上确实生效且 fail-closed：
- `_VcvtSignature`（ops.py:146-183）强制 direction∈{forward,reverse}、part 必须是 int 0、elem_width 必须 16、forward 必须 carrier→f32 RMEM、reverse 必须 f32 RMEM→carrier。**违反任一即拒**（实测原文：vcvt 配 pset(32) 谓词 → `Vcvt: predicate operand 'mask' is produced by 'Pset' at 32-bit mask granularity, but a conversion instruction ... requires a 16-bit mask produced by a pset of the same granularity in this region instance; a bool tensor's shape does not encode the predicate granularity`）。
- `_VstsPack`（ops.py:202-223）强制 carrier 源必须 pack="PK_B32" 且落 bf16 UB。
- 回程不可绕道：`ascend_ub_to_gm` 要求 src.dtype==dst.dtype（实测 `src.dtype=f32 dst.dtype=bf16, reads src.dtype = dst.dtype`），所以 f32 计算结果写回 bf16 GM 必经 reverse vcvt——bf16 e2e 是**真链**，不是装饰。
- fp16 对比：作者程序第一步即拒 `src: f16 does not match f32`（_IN_GMEM_LOAD 只允许 f32/bf16，ops.py:74）；但发射面 TL_DTYPES（emit.py 现版）已含 f16→float16。→ **fp16 缺在指令面与 IR 面，不缺在发射 dtype 表**。

## 2. parse / reader 侧实测（G2 现状）

- **我 task-42 报的 "expected 侧 float32 字面量" 已被并行提交修掉**：tl_reader 现在同样经 `vector_model(...)` 派生（tl_reader.py:405/408/438，PREDICATE_DTYPE 亦 import 自 vector_model 而非本地字面量，:41）。
- 复核任务描述问的 x64-vs-x128：**由真实作者程序 lowered 的等价结果是 bfloat16x128 / extent 128**，与手改文本探针一致；规则出处 = vector_model.py 模块内实测表（float32: x64/64；bfloat16: x128/128；float16: x128/128；并解释早期"extent==tile extent"只是 32-bit 情形的巧合）。
- 更正我上一版草稿的一处推断：x128 **不是**"128 个可寻址 lane 的语义承诺"，而是部署工具链按 2048-bit 寄存器文件做的几何换算；它是否对应真机 16-bit 打包，vector_model 自己标 UNSOURCED，reader 据此只给 INCOMPLETE。**任何"按位宽重算 lane"的实现都只是在复述部署规则，不能自行发明第二种换算。**
- 因此对 G2 的判定要改口径：reader/emit 一致面**在 32-bit 上已同源**；16-bit 上"一致"可达但**认证状态天花板是 INCOMPLETE**，除非设备证据落地。

## 3. G1 真实缺项计数与"前三步能否先换结构代验"

按可分离动作计（bf16 视角，fp16 另加①）：

| # | 缺项 | 类别 | 最小动作 | 阻塞谁 |
|---|---|---|---|---|
| ① | 指令面 fp16：_IN_GMEM_LOAD/_IN_UB_LOAD/_IN_VREG 全线只 f32/bf16（ops.py:51/:56/:71-77） | 需 TF 新签名 | 扩 pattern 与 dtype 校验；**与 S0 合同改动需 architect 定** | fp16 全部 18 例 |
| ② | carrier scope 发射面：emit.py:313-317 只认 smem | 仅缺声明/映射 | _declare 承认 carrier 并给其 TL 侧声明形式（TL 已可 parse 16-bit） | bf16 e2e parse |
| ③ | vcvt/pset16 无 TL_SURFACE+renderer（emit.py TL_SURFACE 6 条；:452 TL_NO_SURFACE） | 需新发射映射 | 两项各加 surface+renderer+expected 派生 | bf16 e2e parse |
| ④ | 16-bit lane/打包硬件证据：vector_model 16-bit=UNSOURCED，reader 上限 INCOMPLETE | 需硬件证据 | validator 设备窗口对照（勿用 CPU 换算冒充） | VERIFIED 状态本身 |

**能做前三步就先拿结构代验吗？可以。** ②③ 闭合后（均不需硬件），bf16full 可 emit+parse+read 得逐项一致、状态 INCOMPLETE——这正好是 G2 能核的形式；用于结构/调度/同步代验合法。**但 G4 必须同时生效：INCOMPLETE≠VERIFIED≠原例数值 PASS。精确计数：40 例全部是 16-bit（AR bf16×17 + fp16×3；MLA fp16×15 + bf16×5）——②③ 只解锁其中 22 个 bf16 例的数值路径上限（且受 ④ 封顶），18 个 fp16 例还额外等 ①；任何例的 VERIFIED 都必须等 ④。**

## 4. 停手条件核查

未触发"必须设备才能继续"：parse-only 全程可推进，卡点①②③④中只有 ④ 属设备窗口，且不阻塞 ②③ 的落地。本报告结论均以退出码/原文为据，可在报告 §5 复跑。

## 5. 复跑命令

```bash
cd /home/s00653124/workspace/TileFoundry_950PR_handoff
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
  /home/s00653124/workspace/envs/tfnpu-v2/bin/python /tmp/tf_fac/run_bf16full2.py   # bf16full lowered OK, 卡 TL_BUFFER_SCOPE_MISMATCH
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
  /home/s00653124/workspace/envs/tfnpu-v2/bin/python /tmp/tf_fac/run_bf16.py        # bf16load/bf16e2e/fp16 三态失败原文
```

## 6. 更正记录（防静默）

- 草稿曾断言"x128 是传输单位换算、G1 只差声明"：按最新工作树实测**更正**为——x128 = 部署 _vreg_lanes(2048//bits) 的寄存器几何（vector_model.py，16-bit UNSOURCED），真实卡点是 carrier scope（②）与 vcvt/pset16 无面（③），另有硬件证据项（④）。
- task-42 §1 的"reader expected 侧是 f32 字面量"：对当时的 tl_reader 成立，**现已被并行提交修复**（tl_reader.py:405/408 同源化），本报告 §2 记录该状态迁移，不改 task-42 历史结论（G3 阈值部分 lead 已按实测定稿）。
