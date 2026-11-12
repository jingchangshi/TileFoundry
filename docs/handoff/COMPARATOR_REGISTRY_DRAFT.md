# COMPARATOR_REGISTRY_DRAFT — G3 登记物说明（task-56）

配套：docs/handoff/COMPARATOR_THRESHOLD_REGISTRY.json（44239 字节，44 条 = 40 权威 + 4 supplemental；registered_at_utc=2026-11-12T23:37:28Z，draft=true）。生成器可复跑：/tmp/tf_fac/gen_registry.py。**本登记物不证明任何原例数值通过。**

## 1. 为什么这样登记

- 阈值不是我们的决定，是**读来的事实**。故每条 case 同时存 `source_forms`（源字面量原文，如 `2 ** -7`）与 `values`（折叠值）：任一方被改写都能被发现，也不强迫下游相信某一种记法。**存双份的正是下面第 3 节的坑。**
- 阈值按 **actual.dtype 选表**：cann_bench_compare.py:785 `_thresholds_for_dtype(output_dtype)`，output_dtype 来自 actual（:548），且 `golden.to(dtype=actual.dtype)`（:787）→ 这就是「f32 代验不能冒充原例」的机制依据：输出 f32 就落 f32 档（2^-13），根本不经过 fp16(2^-10)/bf16(2^-7)。
- 判定式记原码：`:843-845` `mere < threshold and mare < 10.0 * threshold`；small/cancel 另有独立通过位。
- proto.yaml 无 `precision_thresholds` **只影响整数/bool 精确分支**（:712-741、:756-770）；该键仅是可选 MERE 覆盖（:1176-1191），缺省即用表值 → 44 条一律按 dtype 直接登记表条目。
- 身份三件：文件 sha256 + BiShengKernelAgent commit `5496ef6e`（实测 git status 干净）+ 比较器**无 `__version__`**（已 grep 确认）→ 只能这样钉。
- 树内锚点独立钉调用形状：work_package/fixtures/cannbench_contract.py:983/:1006/:1008/:934，其 sha256 与 PACKAGE_MANIFEST.json:175 一致（本次复算通过）。

## 2. 五值原指数表（照原值，未归一化）

顺序固定：threshold / small_value_threshold / small_value_error / cancel_boundary / cancel_zero_threshold。表体 cann_bench_compare.py **82-132**；:133 fallback = float32 档。

| dtype | source_forms（原文） |
|---|---|
| float16 | `2 ** -10` , `2 ** -11` , `2 ** -16` , `2 ** -5` , `2 ** -5` |
| bfloat16 | `2 ** -7` , `2 ** -8` , `2 ** -16` , `2 ** -3` , `2 ** -3` |
| float32 | `2 ** -13` , `2 ** -14` , `2 ** -30` , `2 ** -8` , `2 ** -8` |
| float64 | `2 ** -13` , `2 ** -14` , `2 ** -30` , `2 ** -8` , `2 ** -8` |
| hifloat32 | `2 ** -11` , `2 ** -12` , `2 ** -28` , `2 ** -8` , `2 ** -8` |
| float8_e4m3 | `2 ** -3` , `2 ** -4` , `2 ** -6` , `2 ** -1` , `2 ** -1` |
| float8_e5m2 | `2 ** -2` , `2 ** -3` , `2 ** -5` , **`1.0`** , **`1.0`** |

**float8_e5m2 后两项是 float 常量 `1.0`，不是 `2 ** -n`。** 校验器不得假设五值全为 2 的幂、不得归一化（登记物里 `non_normalisation_note` 也写了这条）。

## 3. 给 task-52 校验器的坑（会咬人）

- **`ast.literal_eval` 读不了这张表**：`2 ** -10` 是 ast.BinOp，literal_eval 直接抛 `ValueError: malformed node ...`（flex-1-g1 踩过、lead 已确认）。必须 **BinOp-aware 折叠**：left 常量 2、op=Pow、right 一元负号。本登记物生成器即如此实现。
- 因此校验器应**独立实现同一折叠、从表文件重新推导**，再与 JSON 的 source_forms+values 双向比对；单向信任 JSON 等于没校验。这也是 JSON 存双份的原因：values 可被折叠复现，source_forms 可被原文比对，任改其一都会红。
- 正例（必须先有）：现登记物对现表 → 44/44 条目一致、三类 sha256 复算通过、字段齐全（本次自检全绿，见第 6 节）。篡改任一 source_form / value / 路径 / sha → 必须红。

## 4. 两条 UNKNOWN（不可省）

1. **S4/S5 运行时真正 import 的是否同一份文件未证**：harness 用裸模块名 `from npu_compare import ...`（:983），实际加载哪份由设备窗口 PYTHONPATH 决定。闭合方式：P0 记录里抓运行时 `npu_compare.__file__` 的 sha256 再对齐本登记物。
2. **无 `__version__`**：身份只能钉 commit + per-file sha256；该仓一旦移动或文件被编辑，全部 threshold_entry 必须重新折叠，不得继续采信。

## 5. 显式声明

- 本文件与 JSON 都是**登记物**：只说明「每个 case 将按哪一档、哪条判定式被评判」，**不证明**任何 case 数值通过；40/40 权威例 execution_status 仍 NOT_RUN，supplemental 仅扩展覆盖。
- **G3 未由本任务闭合**：闭合还需 task-52 的只读校验器（正例 + 篡改必红）。
- 禁止项照抄合同：不改比较器、不改 proto、不写自造 allclose（CONTRACT.md:44）；候选仍须返回原 dtype（CONTRACT.md:18）。

## 6. 本次自检输出（正例证据，可复跑）

```text
wrote docs/handoff/COMPARATOR_THRESHOLD_REGISTRY.json 44239 bytes, 44 case entries
SELF-CHECK entries_consistent: True | mismatched: []
SELF-CHECK top_fields_present: True | missing: []
SELF-CHECK comparator_sha_still_matches: True   # ef2506bb...
SELF-CHECK table_sha_still_matches: True        # e347b921...
SELF-CHECK anchor_sha_still_matches: True       # 026f51a9...
SELF-CHECK case_count: 44 = 40 authoritative + 4 supplemental
SELF-CHECK e5m2 raw preserved: [2 ** -2, 2 ** -3, 2 ** -5, 1.0, 1.0]
```
