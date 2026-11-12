# TF-FACILITIES — 构建/克隆/替换/校验/执行 TIR 的现成设施指南

- 作者：explorer-g1（task-11）。只读调查；验证脚本在 /tmp/tf_fac/（未入仓库）；基线 HEAD=`f341e47`。
- 所有 Python 实测命令：`PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m tilefoundry.cli ...`。

## 1. 作者侧合法 lexical mesh（最重要）

**结论：所有 instruction issue 必须写在 `with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:` 区域内；`T.SimdVF()` 只提供 execution kind，不提供 mesh。** 可通过 parser+analyze+finalize 的最小程序（实测 = /tmp/tf_fac/p2_min_working.py，参照 examples/rmsnorm_950pr/vadd_chain.py:40-61 / m5_vf_capture.py:18-27）：

```python
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

@module(entry="vmove", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class MinMove:
    @func
    def vmove(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:   # ← lexical mesh
            loaded = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
            staged = tf.schedule((x, loaded), op=T.ascend_gm_to_ub())
            out = tf.empty(Tensor[(64,), "f32", (64,), "gmem"])
            return tf.schedule((staged, out), op=T.ascend_ub_to_gm())
```

实测：analyze exit 0；`schedule finalize ...:MinMove /tmp/...` exit 0。完整 vadd 链（含 SimdVF 内 4 条指令）examples/rmsnorm_950pr/vf_vadd.py:44-65 同样 analyze+finalize 全过（finalize 输出 27 行 TIR：`@prim_func` + `Mesh((Topology("aiv",1),),...)` + `T.SimdVF():` + `T.alloc_tensor/T.tensor_view/T.ascend_*`）。

错误三态（本树实测，用于对号入座）：

| 作者写法 | analyze | finalize | 错误原文 |
|---|---|---|---|
| ScheduleOp（tf.schedule op=T.ascend_*）在 Mesh 外 | ✗ | ✗ | `ScheduleOp: T.ascend_gm_to_ub / mesh: None does not match Mesh(('aiv',), ...)`（typeinfer placement 检查） |
| **tf.reshard（generic op）在 Mesh 外** | **✓ 通过** | **✗ convert_hir_to_tir.py:1919** | `staged instruction issue has no lexical mesh` ← **实现者遇到的即此** |
| generic 表达式（x+x）在 SimdVF 内 | ✓ | ✗ | `Binary: this operation declares no execution placement and may not run inside ExecutionScope 'ascend.simdvf'` |
| T.SimdVF() 无 Mesh 包裹 | ✗ | ✗ | `ExecutionRegion: ... requires topology ('aiv',) but no mesh is in force at the region` |

关键陷阱：**typeinfer 只挡 ScheduleOp 的 placement（p1/p4 在 analyze 即拒），generic op（tf.reshard/普通表达式经候选选择降到 ascend 指令）漏过 analyze，到 finalize 的 issue 路径才在 :1919 拒绝**——条件是 convert_hir_to_tir.py:1911-1919 `required is None and mesh is None`（mesh 来自词法 Mesh region，SimdVF 的 ExecutionRegion 不注入 mesh）。修复方向是作者侧补 Mesh 包裹（或 Main-A 在 issue 处显式给 required mesh），不是放宽 typeinfer。

## 2. 构造 TIR：真实签名（src/tilefoundry/ir/tir/stmts.py，全部 dataclass 位置参数）

- `Sequential(body: tuple[Stmt,...])` :24-28（统一 body 槽位，可迭代）
- `LetStmt(var: Var, value: Expr)` :40-46（分配即 `LetStmt(v, Call(AllocTensor(tensor_type=tt), (), type=tt))`）
- `For(induction_var: Var, start: Expr, stop: Expr, step: Expr, body: Sequential, pipeline=None)` :48-65（start/stop/step 都是 Expr，用 `Constant(0, type=i64)` / `ceildiv(...)`，见 attnres/build.py:220-239）
- `While(cond, body)` :68-71；`If(cond, then_body: Sequential, else_body: Sequential)` :74-78
- `MeshScope(mesh: Mesh, binding: Var, body: Sequential)` :81-85；`ExecutionScope(kind: str, body: Sequential)` :88-108（kind=注册名，Ascend 用 `SIMD_VF_KIND="ascend.simdvf"`，targets/ascend/tilefoundry_ascend/ir/execution.py:27；body 必须是 Sequential，__post_init__ 强制 :101-108）
- `Evaluate(callable: Op|..., args: tuple[Expr,...])` :116-122；`Return()` :111-113
- `PrimFunction(name, params: tuple[Var,...], body: Sequential, output_count=1, target=Target, ...)`（ir/tir/prim_function.py:19-40；**output_count 计 TRAILING 参数**，输出参数必须放最后——314e016 修复的缺陷，test_operator_abi_numeric.py:1-9）
- 分配/视图：`AllocTensor(tensor_type=TensorType)`（ir/tir/memory/alloc_tensor.py:22-33）；`TensorView(pointer|offset, dtype=, storage=, layout=, shape=)`（ir/tir/memory/tensor_view.py:20-32）
- 类型：`make_tensor_type(shape, DType.f32, "rmem")`（ir/types/utils）；`TensorType.scalar(DType.i64)`；TIR 层 `Mesh((Topology("aiv", 1),), Layout((1,)), ("core",))`（ir/types/mesh.py:39-59，多层必须 Topology 值）
- 现成完整模板：**targets/ascend/tilefoundry_ascend/attnres/build.py:175-269**（build_attnres_module：Var 定义→alloc→ExecutionScope(compute)→For(双层)→MeshScope→PrimFunction→Module+metadata），照抄改算子即可。

## 3. 克隆与替换（src/tilefoundry/ir/visitor.py）

- `ExprCloner` :360-369：identity-preserving 重写；子节点全部 `is` 同一对象则原样返回（`new is old` 判"没改"）；按源表达式 id memoize（共享输入保持共享；需要两份独立拷贝时**每份各建一个 cloner 实例**）。
- `BindingSubstitutionCloner` :380-420：identity-keyed env 替换 Var（`visit_Var` :394-398 按 `ctx.get(id(expr))`）；MeshRegion 等绑定点会为 body 扩展 env（:410-420）。**没有现成 "freshen 局部 Var" 工具**——fresh 命名用 convert_hir_to_tir.py:448-465 的 `Names(used).fresh(stem, authored)`；局部重写照 test_identity.py:48-75 的 `_replace_vmul`：dataclasses 按字段递归 replace（跳过 type/metadata/annotation/origin），是树内验证过的模式。
- `Module.owns(fn, *, derived=False)`（ir/core/module.py:286-309）：**对象 identity**，结构相等/同名不算；derived=True 沿 `_specialized_from` 链回溯。`entry_function()` :311-328：entry 必须唯一匹配否则 ValueError。`lookup(name)` :276-284 同理。替换 entry 的正确姿势：`dataclasses.replace(module, functions=(new_fn,))`（test_identity.py:68-75）。

## 4. 校验（src/tilefoundry/visitor_registry/verify.py）

- TIR：`verify_prim_function(fn, *, module_fns=Module|Iterable[PrimFunction])` :463-501。传 Module 时自动取 target/topologies 环境（:486-498）。调用样板：targets/ascend/tests_native_mainline/test_identity.py:74、test_native_builders.py:98,148。
- 强制规则（实测触发）：(a) INSIDE 指令必须在 `ExecutionScope("ascend.simdvf")` 内——裸 `Evaluate(Vadd(), (a,b,mask,d))` 拒绝：`Vadd: this operation declares it belongs inside execution region 'ascend.simdvf', but no execution region is in force at it`；(b) 参数 layout 同质（全 ShardLayout 或全非）:504-520；(c) binder 必须 fresh（重复绑定拒绝 :543-552）、"used outside the scope that binds it" :531。
- HIR：`verify_function(fn, *, module=None)` :231。VerifyError 带 type 前缀，按消息匹配稳定。

## 5. 执行（三种入口，能力不同——实测）

- **A. authored HIR Module**：`Module.__call__(*args)` → `evaluator.evaluate(entry_function(), *args)`（ir/core/module.py:330-332；evaluator/interpreter.py:408-424，HIR EvaluatorVisitor，输入在哪就在哪算，CPU torch 实测 OK：vf_vadd authored `M(x)` → Tensor[64]）。
- **B. finalized TIR Module：核心 evaluator 不能跑**——实测 `Module.__call__` 报 `AttributeError: 'PrimFunction' object has no attribute 'evaluation_target'`（interpreter.py 无 PrimFunction/ExecutionScope/For 支持）。别把时间花在这条路上。
- **C. native TIR 的现成 CPU 执行器 = 测试 oracle** `targets/ascend/tests_native_mainline/backend_operators/native_reference.py:181-215 run_module(module, tensors, scalars)`：ABI 派生（codegen/abi.abi_from_module）→ validate_bindings → numpy 执行 For/Sequential/ExecutionScope/MeshScope/LetStmt/Evaluate + GmToUb/UbToGm/Pset/Vld/Vsts/Vmul/Aiv* → 返回 `({output_name: array}, event_log)`。实测 attnres 模块两输入 x,v → y==x*v 且输入不被改写。跑 PrimFunction/Module 用它（或照它抄一个执行器），**不要**用 evaluator.evaluate。

## 6. 打印（src/tilefoundry/inspection/python_printer.py:1249）

`as_script(fn, *, module=None, options=None)`：Module 全 PrimFunction → `tir_module_to_python`（本报告 §1 finalize 输出即该形态）；HIR Module → `_module_to_python`；裸 PrimFunction → `tir_function_to_python`。输出是可再 parse 的 DSL 源（`@module/@prim_func + T.*`），直接用于报告与回归 diff。

## 附：实测命令记录（可复跑）

```bash
P="PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend"
$P python -m tilefoundry.cli analyze /tmp/tf_fac/p2_min_working.py:MinMove            # exit 0
$P python -m tilefoundry.cli schedule finalize /tmp/tf_fac/p5_reshard_no_mesh.py:ReshardNoMesh /tmp/x.py
                                                                                      # error: staged instruction issue has no lexical mesh
$P python -m tilefoundry.cli schedule finalize examples/rmsnorm_950pr/vf_vadd.py:VfVadd /tmp/vf_vadd_finalized.py   # exit 0
```
