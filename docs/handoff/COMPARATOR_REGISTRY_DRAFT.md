# COMPARATOR_REGISTRY_DRAFT — G3 登记物说明（task-56）

配套 docs/handoff/COMPARATOR_THRESHOLD_REGISTRY.json —— **v2 定稿**（sha `4a77755eb039adf9…`、schema `…registry.v2`、draft=true、44 条 = 40 权威 + 4 supplemental、stamp 2026-11-12T23:59:54Z）。**绑定只认这个 sha**：v1 `b25dd485…`（44239B、mtime 07:37:28.990、commit `883117e`）与中途 `8d1e9262…` 都是历史，都记在 `registry_history`（v1 含缺陷一行，未抹）。**本登记物不证明任何原例数值通过。**

## 1. 为什么这样登记

- 阈值不是我们的决定，是**读来的事实**：每条同时存 `source_forms`（原文 `2 ** -7`）与 `values`（0.0078125），任一方被改写都能被发现，也不强迫下游相信某种记法——存双份的理由见 §3.2/§3.3。
- 阈值按 **actual.dtype 选表**（cann_bench_compare.py:785；output_dtype 源自 actual :548；`golden.to(dtype=actual.dtype)` :787）→ 这就是「f32 代验不能冒充原例」的**机制**依据：输出 f32 落 2^-13 档，根本不经过 fp16(2^-10)/bf16(2^-7)。
- 判定式记原码 `:843-845 mere < threshold and mare < 10.0 * threshold`；small/cancel 另有独立通过位。proto.yaml 无 `precision_thresholds` **只影响整数/bool 精确分支**（:712-741、:756-770），该键仅是可选 MERE 覆盖（:1176-1191），缺省即用表值。
- 身份三件 + 树内锚点：文件 sha256、BiShengKernelAgent commit `5496ef6e`（status 干净）、比较器**无 `__version__`**（grep 确认）；cannbench_contract.py:934/:983/:1006/:1008 钉住调用形状，其 sha256 与 PACKAGE_MANIFEST.json:175 一致（本次复算通过）。

## 2. 五值原指数表（照原值，未归一化）

顺序固定 threshold / small_value_threshold / small_value_error / cancel_boundary / cancel_zero_threshold；表体 **82-132**，:133 fallback=float32 档。

| dtype | source_forms（原文） |
|---|---|
| float16 | `2 ** -10` , `2 ** -11` , `2 ** -16` , `2 ** -5` , `2 ** -5` |
| bfloat16 | `2 ** -7` , `2 ** -8` , `2 ** -16` , `2 ** -3` , `2 ** -3` |
| float32 | `2 ** -13` , `2 ** -14` , `2 ** -30` , `2 ** -8` , `2 ** -8` |
| float64 | `2 ** -13` , `2 ** -14` , `2 ** -30` , `2 ** -8` , `2 ** -8` |
| hifloat32 | `2 ** -11` , `2 ** -12` , `2 ** -28` , `2 ** -8` , `2 ** -8` |
| float8_e4m3 | `2 ** -3` , `2 ** -4` , `2 ** -6` , `2 ** -1` , `2 ** -1` |
| float8_e5m2 | `2 ** -2` , `2 ** -3` , `2 ** -5` , **`1.0`** , **`1.0`** |

**e5m2 后两项是 float 常量 `1.0`，不是 `2 ** -n`**：校验器不得假设五值全为 2 的幂、不得重编码为 `2 ** 0`（JSON 的 `non_normalisation_note` 同步写了这条）。

## 3. 校验器必须同时具备的六条 guard（每条都有实测漏检，flex-1-g1 内存副本实证、registry 与表文件零写入）+ 6b 登记物缺陷注记

1. **BinOp-aware 折叠**：`2 ** -10` 是 ast.BinOp，`ast.literal_eval` 抛 `malformed node`。规则 left=2 / op=Pow / right=一元负号。
2. **比数值不比字面写法**：`ast.unparse` 出 `2**(-7)` 而 registry 存 `2 ** -7`，按字符串比会把**完好** registry 判成 44/220 全红（连续两版检查器因此皆错，第三版基线绿才配下结论）。
3. **必须从表文件独立重推导**：协调篡改（fp16 `2**-10→2**-9`，source_forms 与 values 同步改、内部仍自洽）实测单向自洽 **0 红＝漏检**、重推导 **2 红＝抓住**。只读 JSON 等于没校验。
4. **count / id-set 必须绑外部文件，且绑对字段**：丢一条 case（44→43）时上述两类检查**全绿**。v2 的机械绑法（**无需任何改名规则**）：`{source_key} ==` 重读两个源文件所得 44 个 `key` 的集合（44 个互异、逐字节相等，复算 True）；`{source_case_id}` **只能按 operator 分组或按多重集比** —— 实测权威 40 条的源值是 **int 1..20、每个 operator 重复一次**（不是 lead v2 规格里写的 "1…40" 字符串），全表 distinct 只有 24，故全局集合相等虽然可满足但**几乎不证明什么**；全局断言必须用 source_key，不是 source_case_id。（JSON 内 `authoritative_total=40` 只是自我声明，不可替代外部比对。）registry 本地 `case_id`（= `<case>/<case_id>` 拼接）是**派生标识、不是源字段**，不得拿去与源字段比；`source_key` 唯一契约是等于源 `key` 原文，前缀形态（`level4/` vs `supplemental/mla_`）**无语义、不许断言、不许统一、不许再造第三种记法**（`binding_rules.source_key` 已写明）。→ **给 task-52**：v2 定稿后请删掉 suffix 映射与 `prefix=="mla_supplemental"` 断言，改断言 `{source_key}==` 源 key 集合（我实测：这两条在 v2 上仍会通过，但一旦 case_id 改成源字面就会误杀，且它们本身已是 lead 禁止的私有规则）。
5. **字段齐全要显式 guard 且 shape-aware**：删任一字段是裸 `KeyError`（折叠检查不会顺带发现）。本 registry 有**两种合法 shape**（40 条无 `supplemental`、4 条多 `supplemental: true`），统一断言字段集合会把完好件判红。必检顶层 9 键 + 每条 case 5 键 + `threshold_entry{order,source_forms,values}` 且 order 为五字段顺序。
6b. **登记物缺陷——已由 v2 `source_key` 消除**（保留一行历史）：v1 曾把 supplemental 记成 `mla_supplemental/\<s\>`，那是我生成器的自造记法、不对应源文件任何字段（源为 `case_id=\<s\>` + `key=supplemental/mla_\<s\>`）。v2 每条同时存 `source_file/source_key/source_case_id`（逐字节拷贝）并保留 `case_id` 作为 registry 本地标识；`registry_history.v1` 记下旧 sha/commit。→ 校验器不再有不可验证的私有改名层。
7. **身份 guard**：三 sha256 复算 + `line_range` 仍锚住表体 + `repo_head` 等于 BiShengKernelAgent 当前 HEAD；表若移动则重推导整体失败，而非静默沿用旧值。

## 4. 两条 UNKNOWN（不可省）

1. **S4/S5 运行时 import 的是否同一份文件未证**：harness 用裸模块名 `from npu_compare import ...`（:983），实加载哪份由设备窗口 PYTHONPATH 决定。闭合需在 P0 抓运行时 `npu_compare.__file__` 的 sha256 对齐本登记物——**即使 emit/校验器全绿，这一条仍不闭合**。
2. **无 `__version__`**：身份只能钉 commit + per-file sha256；该仓一旦移动或文件被编辑，全部 threshold_entry 必须重新折叠，不得继续采信。

## 5. 显式声明

- 本文与 JSON 都只是**登记物**：说明「每个 case 将按哪一档、哪条判定式被评判」，**不证明**任何 case 数值通过；40/40 权威例 execution_status 仍 NOT_RUN，supplemental 仅扩展覆盖。
- **G3 未由本任务闭合**：闭合还需 task-52 的只读校验器（正例 + §3 六条 guard 各自的负例）。**v2 后 §3.4 不再需要改名规则**；仍按 v1 suffix 口径写的校验器要改成比 `source_key`，否则会把完好 v2 判红（同 §3.2 型错误）。**task-52 全绿也不等于原 40 例可跑**——正向链仍卡在指令面/发射面（task-51/55），dist-pack 静默丢弃属 G1/G2 风险面、不改动本表任何数值。
- 禁止项照抄合同：不改比较器、不改 proto、不自造 allclose（CONTRACT.md:44）；候选须返回原 dtype（CONTRACT.md:18）。

## 6. 本次自检输出（仅**内部**一致性；外部绑定见 §3.3/§3.4）

```text
v2: 51489 bytes, 44 entries | C1 source_key set==sources True(44) | C2 per-operator case_id sets True | C4 dtype per source row True | C6 shapes 2(40x8,4x9) | fields added only [source_file,source_key,source_case_id], removed none
comparator_sha ef2506bb… / table_sha e347b921… / anchor_sha 026f51a9… 复算 True | case_count 44 = 40 权威 + 4 supplemental
e5m2 raw preserved [2 ** -2, 2 ** -3, 2 ** -5, 1.0, 1.0]
```
另证 additive-only：与 `git show 883117e` 的 v1 逐条比，44 条 dtype+threshold_entry+registration_basis **byte-equal**、case_id 集合不变、无字段删除。注：以上均为**内部/绑定**自检，抓不住 §3.3 的协调篡改与 §3.4 的丢条目，不要把本节当成 G3 闭合证据。