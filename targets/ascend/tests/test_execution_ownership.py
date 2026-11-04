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
            # The mesh-bound value, typed exactly as the working direct-TIR test
            # (test_execution_scope_verify.py) types it, so a refusal here cannot be
            # about the mesh binding rather than the case under test.
            Var(name="aiv", type=_tensor(storage="smem")),
            Sequential(body=body),
        ),
        output_count=0,
    )


#: Layers a refusal can come from. A negative is only evidence for a contract when
#: the layer that refused is the layer that STATES it, so every case records which
#: one fired instead of accepting the first exception as a pass.
#:
#:   "operand-pattern"  a declared operand Pattern refused the operand: the
#:                      declaration itself contains the rule, enforced before any
#:                      instruction is admitted.
#:   "shape-pattern"    a shape/layout pattern or projection refused the extent:
#:                      the extent is encoded in the pattern, so this layer carries
#:                      the lane count.
#:   "lane-rule"        verify_operand_lanes refused: the extent rule as a named
#:                      verifier check.
#:   "verifier"         some other registered rule refused (placement, ownership,
#:                      instruction constraints).
#:
#: Anything else is a defect in the FIXTURE, not a pass: it is raised, never
#: counted. That is what stops an unrelated refusal from standing in for the rule.
LAYER_PATTERN = "operand-pattern"
LAYER_SHAPE = "shape-pattern"
LAYER_LANES = "lane-rule"
LAYER_VERIFIER = "verifier"
LAYERS: list = []


def _layer_of(text: str) -> str:
    """A BEST-EFFORT label for the layer that refused, from its message.

    This is a heuristic over the refusal text, and it is labelled as one because the
    honest alternative -- having the verifier report which rule fired structurally --
    is a change to the verifier, not to this fixture. It is deliberately shallow: it
    separates "a pattern refused it" from "a named rule refused it" and, within the
    patterns, whether the message talks about extent at all. It does NOT try to be
    clever, because a clever classifier that guesses wrong would launder a
    wrong-layer refusal into a pass -- which is exactly what this file exists to
    prevent.

    Every refusal prints its first line, so a reviewer sees what actually arrived
    next to the label. If a label disagrees with the text on the first real run, the
    case is re-pinned to the text by hand; the label never overrides the message.
    """
    lowered = text.lower()
    mentions_extent = "lane" in lowered or str(LANES) in text
    if mentions_extent and "does not match" not in lowered:
        return LAYER_LANES
    if "does not match" in lowered and mentions_extent:
        return LAYER_SHAPE
    if "does not match" in lowered or "operand pattern" in lowered or "no variant" in lowered:
        return LAYER_PATTERN
    return LAYER_VERIFIER


def _refusal(function: PrimFunction) -> str:
    """The refusal text, with the layer that produced it appended to LAYERS."""
    try:
        verify_prim_function(function)
    except VerifyError as error:
        text = str(error)
        LAYERS.append(_layer_of(text))
        # Printed always, so the label is never the only thing on record.
        print(f"       refused: {text.splitlines()[0][:150]}")
        return text
    raise AssertionError("expected a refusal; the program was accepted")


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
    layer = LAYERS[before]

    # The dtype contract is stated in TWO places on purpose: the operand pattern on
    # the declaration, and the verifier rule that reads the actual operand type. A
    # refusal from EITHER is the same contract enforced. Being refused by the
    # PATTERN is the better outcome -- the declaration itself carries the rule, so it
    # fires before any instruction is admitted -- so pattern is the expected layer
    # here and the verifier rule is the backstop for operands arriving another way.
    assert layer in (LAYER_PATTERN, LAYER_VERIFIER), (
        f"refused by {layer!r}, which does not encode the element type: {message}"
    )
    assert "i32" in message, message
    if layer == LAYER_VERIFIER:
        assert "computes on" in message, message
    print(f"     layer={layer}")


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
    layer = LAYERS[before]

    # The lane count is asserted on the layer that carries the extent, not merely on
    # the exception. Two layers encode it: the shape/projection pattern (the extent
    # is part of the declaration) and verify_operand_lanes (the named rule).
    assert layer in (LAYER_SHAPE, LAYER_LANES), (
        f"refused by {layer!r}, which does not encode the lane count: {message}"
    )

    # STRICT IS THE DEFAULT: the message is expected to name the declared count.
    # It is relaxed to "says something about lanes" only for the two layers that
    # actually carry the extent, because a correct refusal may describe the extent
    # ("operand shape (32,) is not the 64 lanes ...") in a wording that does not
    # spell the declaration number, and asserting the number would fail a CORRECT
    # refusal. If the layer does not encode the extent, the relaxed form is not
    # acceptable -- the assertion above has already failed the case by then.
    assert "lanes" in message or str(LANES) in message, message
    if str(LANES) not in message:
        print(f"     relaxed: {layer} refused without spelling {LANES}; message says lanes")
    print(f"     layer={layer}")


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
    if LAYERS:
        print("refusal layers seen:", LAYERS)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())