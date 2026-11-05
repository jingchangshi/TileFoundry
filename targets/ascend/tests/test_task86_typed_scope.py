"""task-86: R rewritten TIR must preserve LetStmt/SSA binding scope.

Two named bad controls on a synthetic typed TIR:
  * delete a LetStmt binding but keep a use -> PLAN_TYPED_VERIFY_FAILED;
  * move a use outside the scope that binds it -> PLAN_TYPED_VERIFY_FAILED.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]


def _parts():
    from tilefoundry.ir.core import Call, Constant, Var
    from tilefoundry.ir.tir.memory.copy import Copy
    from tilefoundry.ir.tir.memory.tensor_view import TensorView
    from tilefoundry.ir.tir.prim_function import PrimFunction
    from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, MeshScope, Sequential
    from tilefoundry.ir.types import DType, Layout, Mesh, PointerType, Topology, make_tensor_type
    from tilefoundry.ir.types.storage import StorageKind

    tensor_type = make_tensor_type((8,), dtype=DType.f32, storage=StorageKind.SMEM)
    pointer = PointerType(tensor_type.dtype, tensor_type.storage)
    scalar = make_tensor_type((), dtype=DType.i64, storage=StorageKind.UMAT)
    mesh = Mesh((Topology("thread", 1),), Layout((1,), (1,)))
    bind = Var(type=make_tensor_type((), storage="rmem"), name="m")

    def view(offset):
        return Call(type=tensor_type,
                    target=TensorView(storage=tensor_type.storage, dtype=str(DType.f32),
                                      layout=Layout((8,)), shape=(8,)),
                    args=(Constant(type=pointer, value=offset),))

    bound = Var(name="bound", type=tensor_type)
    let = LetStmt(var=bound, value=view(0))
    use = Evaluate(callable=Copy(), args=(bound, view(32)))
    loop = For(induction_var=Var(name="i", type=scalar),
               start=Constant(type=scalar, value=0),
               stop=Constant(type=scalar, value=1),
               step=Constant(type=scalar, value=1),
               body=Sequential(body=(let, use)))
    scope = MeshScope(mesh=mesh, binding=bind, body=Sequential(body=(loop,)))
    function = PrimFunction(name="scope_probe", params=(), body=Sequential(body=(scope,)), output_count=0)
    return function, let, use, loop, scope


def test_binding_present_is_typed_valid():
    from tilefoundry_ascend.plan.materialize import verify_typed_tir

    function, _let, _use, _loop, _scope = _parts()
    assert verify_typed_tir(function)["valid"] is True


def test_bad_delete_binding_but_keep_use():
    from tilefoundry_ascend.plan.materialize import verify_typed_tir
    from tilefoundry_ascend.plan.resolved_plan import PlanError
    from tilefoundry.ir.tir.stmts import Sequential

    function, _let, use, loop, scope = _parts()
    import dataclasses
    bad_loop = dataclasses.replace(loop, body=Sequential(body=(use,)))
    bad_scope = dataclasses.replace(scope, body=Sequential(body=(bad_loop,)))
    bad = dataclasses.replace(function, body=Sequential(body=(bad_scope,)))
    try:
        verify_typed_tir(bad)
    except PlanError as error:
        assert error.code == "PLAN_TYPED_VERIFY_FAILED", error.code
    else:
        raise AssertionError("deleting the binding should fail typed verification")


def test_bad_use_outside_binding_scope():
    from tilefoundry_ascend.plan.materialize import verify_typed_tir
    from tilefoundry_ascend.plan.resolved_plan import PlanError
    from tilefoundry.ir.tir.stmts import Sequential

    function, let, use, loop, scope = _parts()
    import dataclasses
    loop_only_binding = dataclasses.replace(loop, body=Sequential(body=(let,)))
    bad_scope = dataclasses.replace(scope, body=Sequential(body=(loop_only_binding, use)))
    bad = dataclasses.replace(function, body=Sequential(body=(bad_scope,)))
    try:
        verify_typed_tir(bad)
    except PlanError as error:
        assert error.code == "PLAN_TYPED_VERIFY_FAILED", error.code
    else:
        raise AssertionError("moving a use outside its binding scope should fail")


def main() -> int:
    tests = (test_binding_present_is_typed_valid, test_bad_delete_binding_but_keep_use,
             test_bad_use_outside_binding_scope)
    failed = 0
    for case in tests:
        try:
            case()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s: %s" % (case.__name__, type(error).__name__, error))
        else:
            print("PASS %s" % case.__name__)
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
