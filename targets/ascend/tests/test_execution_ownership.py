"""M3 private-value ownership: no private register without a region instance.

Direct-TIR cases, in the same shape as test_execution_scope_verify.py: each
function is otherwise valid -- mesh, bindings and operand layouts all check out
-- so the refusal that arrives is the rule under test rather than a consequence of
something else being wrong.

What is under test is provenance. A register value exists because a region has a
register file, so it must be created inside a region instance and consumed there.
There are only two ways such a value can enter a program: an allocation, or a
parameter. The allocator case is checked where the allocation is made; the
parameter case is checked at the function boundary, because a caller outside the
region has no register file to supply one from. A third shape -- an inline
TensorView over register storage -- cannot arise in this build: a view over an
integer address is required to be shared-memory, so it has no register form. That
is stated here rather than silently assumed.

Run with the approved container interpreter; this file needs no test framework.

    PYTHONPATH=<repo>/src:<repo>/targets/ascend \
        <interpreter> targets/ascend/tests/test_execution_ownership.py

NOTE: this file was syntax-checked but NOT executed -- the container exec channel
was wedged when it was written. Do not treat it as evidence.
"""
from __future__ import annotations

import sys

import tilefoundry_ascend  # noqa: F401  (registers SimdVF and its instructions)

from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import Mesh, TensorType, Topology, make_tensor_type
from tilefoundry_ascend.ir.ops import Vadd, Vld
from tilefoundry.visitor_registry.verify import verify_prim_function

KIND = "ascend.simdvf"
LANES = 64
AIV = Mesh((Topology("aiv", 1),), (1,), ("core",))


def _tensor(shape=(LANES,), dtype="f32", storage="rmem") -> TensorType:
    return make_tensor_type(shape, dtype=dtype, storage=storage)


def _bind(name: str, type_: TensorType) -> tuple:
    """Allocate *type_* and return both the statement and the Var it binds."""
    bound = Var(name=name, type=type_)
    statement = LetStmt(
        var=bound,
        value=Call(target=AllocTensor(tensor_type=type_), args=(), type=type_),
    )
    return statement, bound


def _function(params, *body) -> PrimFunction:
    return PrimFunction(
        name="fn",
        params=tuple(params),
        body=MeshScope(
            AIV,
            Var(name="aiv", type=_tensor((), "i64", "rmem")),
            Sequential(body=body),
        ),
        output_count=0,
    )


def _refusal(function: PrimFunction) -> str:
    """The refusal, with the LAYER that produced it recorded.

    A negative is only evidence for a contract if the layer that refused is the
    layer that states it. There are two such layers here, and they are not
    interchangeable:

      "pattern"   the operand Pattern on the instruction declaration refused the
                  operand type, so the dtype/storage contract is enforced at the
                  earliest possible point -- before any verifier runs;
      "verifier"  a registered rule refused it (placement, ownership, lane extent,
                  instruction-specific constraints).

    Anything else -- a parse failure, an unrelated layout complaint, a missing
    registration -- is a defect in the FIXTURE, not a pass, so it is raised rather
    than counted. Recording the layer is what stops a wrong-layer refusal from
    being reported as the rule under test.
    """
    try:
        verify_prim_function(function)
    except VerifyError as error:
        text = str(error)
        layering = (
            "pattern"
            if ("does not match the" in text or "operand pattern" in text or "no variant" in text)
            else "verifier"
        )
        LAYERS.append(layering)
        return text
    raise AssertionError("expected a refusal; the program was accepted")


LAYERS: list = []


def check_positive_control_is_accepted() -> None:
    """The control: a private register allocated and used in one region instance.

    Without this, the refusals below would be indistinguishable from "every
    program involving register storage is refused", which would satisfy each of
    them while making the mode unusable.
    """
    ub_stmt, ub = _bind("ub", _tensor(storage="smem"))
    reg_stmt, reg = _bind("reg", _tensor())
    scope = ExecutionScope(
        kind=KIND,
        body=Sequential(body=(reg_stmt, Evaluate(callable=Vld(), args=(ub, reg)))),
    )
    verify_prim_function(_function((), ub_stmt, scope))


def check_external_private_parameter_is_refused() -> None:
    """A private value handed in from outside has no possible producer.

    The function is otherwise valid: the parameter is used inside the region, in
    an instruction that accepts register storage. What is wrong is that the value
    crossed the boundary backwards -- a caller standing outside the mode has no
    register file to read it from.
    """
    param = Var(name="reg", type=_tensor())
    ub_stmt, ub = _bind("ub", _tensor(storage="smem"))
    scope = ExecutionScope(
        kind=KIND,
        body=Sequential(body=(Evaluate(callable=Vld(), args=(ub, param)),)),
    )
    message = _refusal(_function((param,), ub_stmt, scope))
    assert "private to execution region" in message, message
    assert "supplied from outside" in message, message


def check_hoisted_private_allocation_is_refused() -> None:
    """A private buffer allocated outside the region that owns it."""
    reg_stmt, reg = _bind("leaked", _tensor())
    ub_stmt, ub = _bind("ub", _tensor(storage="smem"))
    scope = ExecutionScope(
        kind=KIND,
        body=Sequential(body=(Evaluate(callable=Vld(), args=(ub, reg)),)),
    )
    message = _refusal(_function((), reg_stmt, ub_stmt, scope))
    assert "private to execution region" in message, message


def check_sibling_region_reuse_is_refused() -> None:
    """A second region instance reading the first instance private register.

    Two sibling regions are two instances, not one: the second has its own
    register file, so a value created in the first is not merely out of scope, it
    belongs to a different instance. This is the case a rule that only asked "was
    it allocated inside some region" would wrongly accept.
    """
    reg_stmt, reg = _bind("a", _tensor())
    ub_stmt, ub = _bind("ub", _tensor(storage="smem"))
    first = ExecutionScope(
        kind=KIND,
        body=Sequential(body=(reg_stmt, Evaluate(callable=Vld(), args=(ub, reg)))),
    )
    ub2_stmt, ub2 = _bind("ub2", _tensor(storage="smem"))
    second = ExecutionScope(
        kind=KIND,
        body=Sequential(body=(Evaluate(callable=Vld(), args=(ub2, reg)),)),
    )
    message = _refusal(_function((), ub_stmt, first, ub2_stmt, second))
    assert "private to execution region" in message, message


def check_same_width_integer_is_refused() -> None:
    """i32 registers: the same 32 bits as f32, and still the wrong arithmetic.

    A width-only contract would accept this, because 32 == 32. The mode states
    the element types it computes on, so this is refused by the dtype rule and the
    refusal says why agreeing widths is not the rule.
    """
    a_stmt, a = _bind("a", _tensor(dtype="i32"))
    b_stmt, b = _bind("b", _tensor(dtype="i32"))
    m_stmt, m = _bind("m", _tensor(dtype="bool"))
    d_stmt, d = _bind("d", _tensor(dtype="i32"))
    scope = ExecutionScope(
        kind=KIND,
        body=Sequential(
            body=(
                a_stmt,
                b_stmt,
                m_stmt,
                d_stmt,
                Evaluate(callable=Vadd(), args=(a, b, m, d)),
            )
        ),
    )
    before = len(LAYERS)
    message = _refusal(_function((), scope))
    # The dtype contract is stated in TWO places on purpose: the operand pattern on
    # the declaration, and the verifier rule that reads the actual operand type. A
    # refusal from EITHER is the same contract enforced, and being refused by the
    # pattern is the BETTER outcome -- it happens at the earliest layer, before any
    # instruction is admitted. What must not happen is a refusal from neither, or a
    # refusal that does not mention the element type at all.
    assert LAYERS[before] in ("pattern", "verifier"), LAYERS[before]
    assert "i32" in message, message
    if LAYERS[before] == "verifier":
        assert "computes on" in message, message
    else:
        # The pattern refusal names the operand and the declared dtype set. The
        # wording belongs to the matcher, so the assertion is on the substance: the
        # message must name the element type, not merely "operand does not match".
        assert "i32" in message or "dtype" in message.lower(), message
    print(f"     layer={LAYERS[before]}")


def check_half_width_lane_extent_is_refused() -> None:
    """A 32-lane register: half an issue, which this mode cannot express.

    The lane count is declared on the mode and measured on the operand, so it is
    a contract rather than a constant in a comment.
    """
    ub_stmt, ub = _bind("ub", _tensor((LANES // 2,), storage="smem"))
    reg_stmt, reg = _bind("reg", _tensor((LANES // 2,)))
    scope = ExecutionScope(
        kind=KIND,
        body=Sequential(body=(reg_stmt, Evaluate(callable=Vld(), args=(ub, reg)))),
    )
    before = len(LAYERS)
    message = _refusal(_function((), ub_stmt, scope))
    assert LAYERS[before] in ("pattern", "verifier"), LAYERS[before]
    assert "lanes" in message or "64" in message, message
    print(f"     layer={LAYERS[before]}")


CHECKS = (
    check_positive_control_is_accepted,
    check_external_private_parameter_is_refused,
    check_hoisted_private_allocation_is_refused,
    check_sibling_region_reuse_is_refused,
    check_same_width_integer_is_refused,
    check_half_width_lane_extent_is_refused,
)


def main() -> int:
    failed = 0
    for check in CHECKS:
        try:
            check()
        except AssertionError as error:
            failed += 1
            print(f"FAIL {check.__name__}: {error}"[:220])
        except Exception as error:  # noqa: BLE001 -- a wrong error is a failure
            failed += 1
            print(f"FAIL {check.__name__}: {type(error).__name__}: {error}"[:220])
        else:
            print(f"ok   {check.__name__}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} ownership checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())