# WP5-BF16-RECON — task-51 的只读侦察（carrier/TL 面/carrier 形态）

explorer-g1 应 lead 指派做 task-51 前置侦察（task-51/52 本体 required_role=implementer，我认领被 TEAM_ROLE_MISMATCH 拒，已回报 lead）。只读，未改任何生产文件；HEAD=f341e47，工作树并发中（本文按 **emit.py mtime 06:15:52、tl_reader 03:26、vector_model 03:22** 的现状核行号；implementer 若再改 emit，行号需重核）。探针：/tmp/tf_fac/probe_carrier_form.py、/tmp/tf_fac/bf16full.py、/tmp/tf_fac/run_bf16full2.py。

## A. carrier scope 在 emit 支持域的判定位置（精确到行）

拒绝链是**两段**，改的时候两段都要动：

1. **误分类点（根因）**：`_bind_definition` 里 AllocTensor 分支（emit.py:645）——
   `self._declare(name, "global" if storage == "gmem" else "shared", ...)`
   storage=="carrier" 不等于 "gmem" → 被强制登记成 **"shared"（UB tile）**。carrier 在这里就没有自己的类别，"lives in carrier" 那句拒绝消息的语义是"它想 declare 成 shared 但 storage 是 carrier"。
2. **显式拒绝点**：`_declare`（emit.py:580-586）——`if scope == "shared": if storage != "smem": raise TL_BUFFER_SCOPE_MISMATCH`。实测拒绝原文：`carr is declared as a UB tile but lives in carrier`。
3. 对照参考：rmem 的正确通路在 emit.py:639-644（`if storage == "rmem"` → registers + `_note_dtype`，**不进 _declare、不占 buffer 表**）；carrier 的承认动作应当类似（register-private、不占 UB 地址表 :648-652 的 _addresses、不进 UB 字节预算）。
4. 合同依据（lead 引的降低端，已核实为真）：`src/tilefoundry/passes/transforms/convert_hir_to_tir.py:273-283` —— RMEM 与 CARRIER 同为 mode-private register，`Call(AllocTensor(...))` 直接 bind、**不取 offsets、不地址化**；:284-287 只允许 SMEM/GMEM 走地址化分支。→ lowered 侧从不给 carrier 算地址，emit 侧的 "shared" 映射是唯一错配点。

## B. vcvt / pset16 在部署 build 的 TL 面（实测，site-packages 0.1.15+cpu.git0e687c17）

**关键更正性发现：部署面上不存在 pset16 函数**（grep 整个 tilelang/ascend/ 零命中）。TF 的 `ascend_pset16(elem_width=16)` 的部署对应物就是同一个 `pset`：

- `tilelang/ascend/language/simd.py:111-117`：`def pset(elem_width: int, dist: str = "PAT_ALL")` → `call_intrin("boolx256", tl.simd.pset, elem_width, dist)`。**elem_width 是运行时整数参数**，16/32 同函数；返回恒 boolx256（与 vector_model.PREDICATE_DTYPE 一致）。→ 接线动作 = TL_SURFACE 加一条 `"pset16": "simd.pset"`，renderer 发 `T.simd.pset(16)`，**不是新算力、也不是找不存在的 pset16**。
- `simd.py:1209-1217`：`def vcvt(src, target_dtype: DType, mask=None, round="ROUND_R", sat=True, part=0, mode="MODE_ZEROING")`。**注意与 TF 侧签名的形状差**：部署 vcvt 没有 `direction`/没有 src/dst 双 buffer——它接收一个**已类型化的 VReg**（`src.dtype` 携元素型），目标由 `target_dtype` 字符串决定（:1276-1277 `src_dtype = str(src.dtype).split("x", 1)[0]`）。TF 的 direction=forward/reverse 对应关系：
  - forward（bf16→f32）：`vcvt(v_bf16vec, target_dtype="float32", ...)`
  - reverse（f32→bf16）：`vcvt(v_f32vec, target_dtype="bfloat16", ...)`
- 元素宽度/粒度：vcvt **不吃 elem_width 参数**；转换谓词粒度由 `mask` 表达——`simd.py:1270-1274`：mask=None 时默认 `_default_mask_bits(min(_mask_bits(src.dtype), _mask_bits(target_dtype)))`，即 **min(32,16)=16** 自动成立。→ 我们要求的 `pset(16)` 谓词与部署默认规则**同源**，TF 显式传 p16 是对该事实的复述，renderer 传 `mask=p16` 即可（sat/part 默认值与 ops.py:146-183 的 part=0 合同一致，round/sat 是否暴露成 TF 属性是独立决策，勿顺手塞进本包）。
- UNPK_B16 侧（**本节初稿推断有误，已按 parse 实测更正**，见下节 C）：`vld(addr, dist="NORM")`（simd.py:324）合法 dist 含 UNPK_B16（:315）。**实测：`T.simd.vld(ub_bf16[0], dist="UNPK_B16")` 返回 `bfloat16x128`（不是 f32！），且发射的 access_ptr extent=64**；f32 只在下一步 `vcvt(v, mask=pset(16))` 之后出现（`float32x64`）。所以 renderer 的 result dtype 序列必须按 parse 事实写，不能按"widening 即 f32"直觉。
- 参考锚点（TL_SURFACE 的既有条目格式）：emit.py:73-83，已含 event 面 `flag_set→event.set_flag / flag_wait→event.wait_flag`（说明 06:15 版已接 sync 面；vcvt/pset16 仍缺）。缺条目时命中 emit.py:754 `TL_NO_SURFACE`。
## C. 部署面 16-bit 发射形态（实测 parse 原文，非文档转述）

同一 64 元素 bf16 UB，两种链路各自 parse 成功后的**实际发射**（/tmp/tf_fac/probe_tl_forms2.py，source set_env.sh 后 parse-only，未 compile/未开设备）：

```text
# forward: 读 + 转 f32 + 存
v: T.bfloat16x128 = T.tl.simd.vld(T.access_ptr(ub[0], 64, 1), "UNPK_B16")
m: T.boolx256     = T.tl.simd.pset(16, "PAT_ALL")
o: T.float32x64   = T.tl.simd.vcvt(v, m, "PART_EVEN", "MODE_ZEROING")
T.tl.simd.vsts(T.access_ptr(ub[0], 128, 2), o, m, "NORM_B32")

# round-trip: f32 再转回 bf16（reverse）
r: T.bfloat16x128 = T.tl.simd.vcvt(o, m, "ROUND_R", "RS_ENABLE", "PART_EVEN", "MODE_ZEROING")   # mask=pset(32)
```

要点（每条都会影响 renderer/expected 的正确性）：
1. **vcvt 的 intrinsic 发射形态不带 target_dtype**：target 被解析成返回类型，参数按位置展开——forward 是 `(v, mask, "PART_EVEN", "MODE_ZEROING")`，reverse（带 sat/round）是 `(o, mask, "ROUND_R", "RS_ENABLE", "PART_EVEN", "MODE_ZEROING")`。→ parsed IR 里**没有 `direction` 字面量**；TF 的 direction 只能由 (src.dtype, result.dtype) 对恢复。reader 比较两侧时须按这个真实形状来。
2. **pset16 的正确发射形态 = `T.tl.simd.pset(16, "PAT_ALL")`**（同一条 op 名 tl.simd.pset）。确认：部署 ascend 语言模块**没有 pset16 函数**（grep 整个 tilelang/ascend 零命中）→ TL_SURFACE 里 "pset16" 应映射 `simd.pset`，不是找一个不存在的函数名。
3. **extent 单位是混合的**（重要）：vld(UNPK_B16) 发射 access_ptr extent=**64**（=元素数），而同一 buffer 上的 vsts 发射 extent=**128**。→ 现在 codegen/vector_model.py 的表（按 buffer dtype 给一个 extent）**不能覆盖 per-instruction/per-dist 的真实差异**；implementer 需按 parse 出的分布给 vld/vcvt/vsts 各自的 extent 派生规则，别假设一条全局规则。（本侦察只报事实，不替 vector_model 定规则。）
4. `vcvt(bf16→bf16)` 会 raise `Unsupported vcvt conversion bfloat16->bfloat16`（simd.py:1423）→ direction 必须由 target dtype 决定，与 ops.py 的显式 direction 合同一致。
5. **部署面不替你把关语义**：把 `float32x64` 直接 `vsts` 进 bf16 UB **parse 成功**（第 1 段示例即如此）→ 部署 surface 不校验 src/缓冲 dtype 一致性。若 TF 的 emitter/reader 需要"转换必经 vcvt"这条合同，它只能在两侧各自派生并比较 expected 来维持；别误以为 reader 的 observed 侧会自动发现这种缺陷。

## D. 真实 bf16 程序在 convert_hir_to_tir.py:273-283 之后的 carrier 实际类型形态（实测）

lowered 入口逐 binder dump（/tmp/tf_fac/probe_carrier_form.py）：

```text
('base',   'bf16',  'SMEM',    'TensorView')   # UB tile（有地址，进预算）
('carr',   'bf16',  'CARRIER', 'AllocTensor')  # mode-private register（无 offsets）
('m16',    'bool',  'RMEM',    'AllocTensor')
('v',      'f32',   'RMEM',    'AllocTensor')
('m32',    'bool',  'RMEM',    'AllocTensor')
('acc',    'f32',   'RMEM',    'AllocTensor')
('carr2',  'bf16',  'CARRIER', 'AllocTensor')
('stored', 'bf16',  'SMEM',    'TensorView')
```

发射指令与属性（同次 dump）：`GmToUb{} → Vld{dist:UNPK_B16} → Pset16{elem_width:16} → Vcvt{direction:forward,part:0,elem_width:16} → Pset{elem_width:32} → Vadd{} → Vcvt{direction:reverse,part:0,elem_width:16} → Vsts{pack:PK_B32} → UbToGm{}`。

要点：
- carrier 在 TIR 里的真实形态 = `Call(AllocTensor(tensor_type=TensorType(dtype=bf16, storage=CARRIER)), (), type=同一 TensorType)`，**无 offsets、无地址**（offsets 元组实测 None/空），与 :273-283 的 mode-private 合同一致；**它不是 TensorView、不与任何 UB 地址共享**，所以"不占 UB 预算"在降低端本来就是事实，缺的只是 emit.py:645 的 scope 分类（"gmem"→"global" 否则→"shared"，没有 register 支路）。
- 两个 CARRIER 都叫 carr/carr2，都在 SimdVF **内部**（ExecutionScope 内层），与 ops.py:342-380 "carrier 只在同 region 内存在"的语义自洽。
- 因此承认 carrier 的最小改动面是（顺序即建议实现顺序）：emit.py:645（_bind_definition AllocTensor 分支的 scope 三分类，carrier→register 类）+ :580-586（_declare 的 shared 校验保持不变，carrier 不再进这里）+ registers 表/渲染里给 carrier 一个合法声明（参考 :637-643 的 rmem 通路），并保证不落入 _declare 内的 _addresses/UB 预算路径（emit.py:589-592）。

## E. vsts 的 pack 面（实测，直接影响 renderer 设计）

TF 侧 `_VstsPack`（ir/ops.py:202-223）要求 carrier 源必须 `pack="PK_B32"`。部署面**能表达 pack**：`vsts(addr, src, mask=None, dist="NORM_B32", extent=None)`（simd.py:457），且 extent 由 dist 派生（:469-477）：`PK_` → lanes//2、`PK4_` → lanes//4、`ONEPT_` → 1，默认 = lanes。实测 parse（/tmp/tf_fac/probe_tl_pk.py）：

```text
T.simd.vsts(ub[0], r, m, dist="PK_B32")
  → T.tl.simd.vsts(T.access_ptr(ub[0], 64, 2), r, m, "PK_B32")     # extent=64
默认（不写 dist）→ T.access_ptr(ub[0], 128, 2), ..., "NORM_B32")    # extent=128
```

→ renderer 必须把 TF 的 `pack` 属性映射到 `dist="PK_B32"`（而不是丢给默认 NORM_B32），且此时 access_ptr extent 是 64。**这同时给 §C 第 3 点一个可执行的规则形式：extent 不是全局常数，而是 (instruction, dist) 的函数，其派生规则完全在部署侧（vld:401-402 UNPK_→lanes//2；vsts:474-477 PK_→lanes//2），TF 侧应当复用同一推导而非另立表——否则 expected 与 observed 会从两个不同真源出发。**

## F. 侦察合规性与已知限制（如实标注）

- 探针性质：parse-only。除 §A/§D 走 TF 正常入口（checked_program→analyze→finalize，无 TL 参与）外，§B/§C/§E 直接对部署 TL 面做 parse（`@T.prim_func` 执行 + 读 `body`），未调 `tilelang.compile`、未产 aibin、未 launch kernel、未打开任何设备节点。
- 合规：§B/§C/§E 初跑在 `source set_env.sh` 之后，日志出现 `torch_npu/utils/collect_env.py: UserWarning`，即该进程里 torch_npu 被间接加载（`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 挡不住显式 `import tilelang.ascend.language` 的路径）。我未在其上分配 npu 张量/未调设备，但为守住纪律已**在不 source CANN env、并加 meta_path 拦截 torch_npu 的条件下复跑**（/tmp/tf_fac/probe_tl_forms_clean.py）：**发射形态与 §C/§E 逐条字节一致**（vld→bfloat16x128/extent 64；vcvt forward→float32x64；vcvt reverse→bfloat16x128；vsts 默认 extent=128 vs dist="PK_B32" 时 extent=64），且 `torch_npu in sys.modules: False`。→ 本节全部形态结论**不依赖 CANN env 与 torch_npu**；§A/§D 从一开始就带拦截器。
- 本文件是**侦察报告**，不是 task-51 的实现交付；task-51/52 的 required_role=implementer，我认领被 TEAM_ROLE_MISMATCH 拒。§A/§B/§C/§D/§E 给出的坐标与形态即为 lead 要求的"file:line + 可复跑探针"。

## G. 复跑命令

```bash
cd /home/s00653124/workspace/TileFoundry_950PR_handoff
# §A/§D：carrier 的真实 lowered 形态与 emit 拒绝点（带 meta_path 拦截，无 torch_npu）
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
  /home/s00653124/workspace/envs/tfnpu-v2/bin/python /tmp/tf_fac/probe_carrier_form.py
# §B/§C/§E：部署 TL 面 16-bit 形态（parse-only；注意 F 节的 torch_npu 说明）
env -u ASCEND_HOME_PATH -u ASCEND_OPP_PATH -u BISHENG_HOME -u LD_LIBRARY_PATH \
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/s00653124/workspace/envs/tfnpu-v2/bin/python /tmp/tf_fac/probe_tl_forms_clean.py   # 带 torch_npu 拦截器
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/s00653124/workspace/envs/tfnpu-v2/bin/python /tmp/tf_fac/probe_tl_pk.py
```