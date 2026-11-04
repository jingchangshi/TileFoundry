"""The execution scope must admit no call it cannot check.

These are direct-TIR cases, not authored programs: a lowered program is not the
only way a scope reaches the verifier, and a rule that only holds for what the
builder emits is a rule about the builder rather than about the IR.

Each case is otherwise valid -- the mesh, the bindings and the operand layouts
all check out -- so the refusal that arrives is the one under test and not a
consequence of something else being wrong.

Run with the approved container interpreter; this file needs no test framework.
"""
from __future__ import annotations

import tilefoundry_ascend  # noqa: F401  (registers the SimdVF surface)

from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.tir.arith import Binary
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import Mesh, TensorType, Topology, make_tensor_type
from tilefoundry.visitor_registry.verify import verify_prim_function

AIV = Mesh((Topology("aiv", 1),), (1,), ("core",))


def _tensor() -> TensorType:
    return make_tensor_type((64,), dtype="f32", storage="smem")


def _bound_function(*statements) -> tuple[PrimFunction, Var]:
    """A function holding one allocated tensor, with the rest of the body given."""
    held = Var(name="held", type=_tensor())
    allocation = LetStmt(var=held, value=Call(target=AllocTensor(tensor_type=_tensor()), args=(), type=_tensor()))
    body = Sequential(body=(allocation, *statements))
    return PrimFunction(name="fn", params=(), body=body, output_count=0), held


def _refusal(function: PrimFunction) -> str:
    try:
        verify_prim_function(function)
    except VerifyError as error:
        return str(error)
    raise AssertionError("expected a refusal; the program was accepted")


def check_computed_callable_inside_mode_is_refused() -> None:
    """A computed instance target is a Call, and a Call may not be entered in a mode.

    The shape matters: an instruction selected at run time arrives on the
    retained-call route rather than the direct-operation one, so a guard written
    as a list of the other callable types leaves it a way through.
    """
    function, held = _bound_function()
    computed = Call(target=AllocTensor(tensor_type=_tensor()), args=(), type=_tensor())
    scope = ExecutionScope(
        kind="ascend.simdvf",
        body=Sequential(body=(Evaluate(callable=computed, args=(held,)),)),
    )
    function.body = Sequential(body=(*function.body.body, MeshScope(AIV, Var(name="aiv", type=_tensor()), Sequential(body=(scope,)))))
    message = _refusal(function)
    assert "callee verification that knows the mode" in message, message
    assert "ascend.simdvf" in message, message


def check_declared_operation_reaches_placement() -> None:
    """A declared operation must be judged on its placement, not on being a callable.

    This is the positive side of the classifier. A direct operation is the one
    callable this walker can actually decide inside a mode, so it must pass the
    callable test and be answered by verify_op_placement. If the classifier
    refuses it first, no native inside-mode operation can ever run -- the
    placement rule would be unreachable for exactly the operations it exists for.

    The operation here declares no placement, so the expected verdict is the
    placement refusal, and the assertion is on which refusal arrived.
    """
    function, held = _bound_function()
    scope = ExecutionScope(
        kind="ascend.simdvf",
        body=Sequential(body=(Evaluate(callable=Binary(kind=BinaryKind.ADD), args=(held, held, held)),)),
    )
    function.body = Sequential(body=(*function.body.body, MeshScope(AIV, Var(name="aiv", type=_tensor()), Sequential(body=(scope,)))))
    message = _refusal(function)
    assert "callee verification that knows the mode" not in message, message
    assert "declares no execution placement" in message, message


def check_undeclared_kind_is_refused() -> None:
    """An unknown kind fails closed rather than being treated as unconstrained."""
    function, held = _bound_function()
    scope = ExecutionScope(
        kind="ascend.not_a_mode",
        body=Sequential(body=(Evaluate(callable=AllocTensor(tensor_type=_tensor()), args=(held,)),)),
    )
    function.body = Sequential(body=(*function.body.body, MeshScope(AIV, Var(name="aiv", type=_tensor()), Sequential(body=(scope,)))))
    message = _refusal(function)
    assert "undeclared kind" in message, message


def check_scope_outside_any_mesh_is_refused() -> None:
    """A scope inside the function body but outside every MeshScope has no domain."""
    function, held = _bound_function()
    scope = ExecutionScope(
        kind="ascend.simdvf",
        body=Sequential(body=(Evaluate(callable=AllocTensor(tensor_type=_tensor()), args=(held,)),)),
    )
    function.body = Sequential(body=(*function.body.body, scope))
    message = _refusal(function)
    assert "no MeshScope encloses it" in message, message


def main() -> int:
    checks = (
        check_declared_operation_reaches_placement,
        check_computed_callable_inside_mode_is_refused,
        check_undeclared_kind_is_refused,
        check_scope_outside_any_mesh_is_refused,
    )
    failures = []
    for check in checks:
        try:
            check()
        except AssertionError as error:
            failures.append(f"{check.__name__}: {error}")
        else:
            print(f"ok   {check.__name__}")
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"{len(checks) - len(failures)}/{len(checks)} checks passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())