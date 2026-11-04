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

COVERAGE, stated precisely so it is not over-read: the ownership rule has ONE
constructible refusal (a private allocation hoisted outside its region) plus the
external-parameter case. The sibling-region case is included but is NOT a second
ownership negative -- the lexical binder refuses it first, and the case asserts that
openly. Do not describe this file as four independent ownership controls.

WHAT THE FIRST REAL RUN MUST SHOW. This file has never executed. When it does, the
run is only evidence if it reports:

  * the interpreter identity, printed by the driver, naming the conda env of record;
  * the positive control as ok AND ACCEPTED -- this is the load-bearing line. A run
    where the positive control fails, or where it is never reached because an
    earlier case raised, proves nothing: every negative below can pass for free if
    the fixture refuses everything;
  * each negative naming the layer that refused, with the layer text printed next to
    it, so a reviewer can see the rule that fired rather than trusting the label;
  * the closing layer summary line, showing which layers were actually exercised.

A negative that refuses for the wrong reason is a FAILED check here, not a near
miss: the assertions pin both the message and the layer. If the first run shows a
refusal arriving from a layer that does not encode the rule, the fixture is wrong
and must be re-pointed -- not the assertion widened.

Run with the interpreter of record, inside a container; this file needs no framework.

    PYTHONPATH=<repo>/src:<repo>/targets/ascend \
        python targets/ascend/tests/test_execution_ownership.py
"""
from __future__ import annotations

import sys

import tilefoundry_ascend  # noqa: F401  (registers SimdVF and its instructions)

from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import DType, Mesh, TensorType, Topology, make_tensor_type
from tilefoundry_ascend.ir.ops import Vadd, Vld
from tilefoundry.visitor_registry.verify import verify_prim_function

KIND = "ascend.simdvf"
LANES = 64
AIV = Mesh((Topology("aiv", 1),), (1,), ("core",))


def _tensor(shape=(LANES,), dtype="f32", storage="rmem") -> TensorType:
    """A tensor type built with REAL DType descriptors, not dtype name strings.

    ``make_tensor_type`` normalizes ``storage`` from a string but NOT ``dtype``: it
    stores whatever it is handed, and ``TensorType.__post_init__`` only touches
    storage and shape. So ``make_tensor_type(..., dtype="f32")`` produces a type
    whose ``dtype`` is the STRING "f32", which compares unequal to ``DType.f32`` and
    has no ``.name``. A fixture built that way is not testing the rules it names:
    a dtype-reading check sees ``None`` and refuses for the wrong reason, and a
    pattern expecting ``DType.f32`` never matches.

    Resolving here keeps every case honest. It is also why the positive control is
    the load-bearing case: it is the one that would have caught this, because a
    string-dtype fixture refuses everything and passes every negative for free.
    """
    return make_tensor_type(shape, dtype=DType.from_name(dtype), storage=storage)


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


def check_sibling_region_reuse_cannot_arise() -> None:
    """A second region instance reaching the first instance private register.

    This is recorded as a REFUSAL THAT COMES FROM A DIFFERENT RULE, deliberately,
    because the alternative -- advertising it as an ownership control -- would be
    claiming coverage that is not there.

    Two sibling regions are two instances, so intuitively the second should be
    refused for reading another instance register file. It cannot be constructed: a
    private value is only visible where it is bound, every binding here is scoped to
    its own Sequential, and the only way to make a value visible inside the second
    region is to bind it OUTSIDE both -- which is the hoisted-allocation case, and
    that is refused at the allocation before either region is reached.

    So the lexical binder refuses first, and the ownership question never arises.
    That is a fact about the LANGUAGE, not a gap being papered over, and the case is
    kept because it is the assertion that would notice if it stopped being true: if
    the binder one day admitted an out-of-scope Var, this case would report the
    ownership refusal instead of the binder one and FAIL, which is exactly what we
    would want to see.

    Consequence to keep in mind when reading the coverage: the ownership rule has
    ONE constructible negative (the hoisted allocation), plus the external-parameter
    case. It does not have two independent ones.
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
    assert "outside the scope that binds it" in message, message
    assert "private to execution region" not in message, message
    print("     refused by the lexical binder, not the ownership rule")


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
    # the declaration, and verify_operand_lanes, which reads the actual operand type.
    # A refusal from EITHER is the same contract enforced, so both are accepted here.
    #
    # WHICH ONE FIRES FIRST is decided by the walker order, not by preference: on the
    # direct-operation route the operand rules run BEFORE the matcher, so the
    # verifier refusal is the one that arrives and the pattern is the backstop for
    # operands reaching the matcher by another route. I am recording that rather than
    # claiming the pattern fires first, which would be a nice story about the earliest
    # possible point and not what the code does.
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
    # the exception. Two layers encode it -- the shape/projection pattern and
    # verify_operand_lanes -- and by the same walker order as above the named rule
    # is the one expected to arrive on this route.
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
    check_sibling_region_reuse_cannot_arise,
    check_same_width_integer_is_refused,
    check_half_width_lane_extent_is_refused,
)


def _refuse_device_stack() -> None:
    """Stop if the NPU stack is loaded. Nothing here needs a device.

    This file is pure IR construction and verification, so a loaded torch_npu can
    only have arrived by accident -- and device access is forbidden while the host
    filesystem wedge is under investigation.
    """
    import sys

    if "torch_npu" in sys.modules:
        raise SystemExit(
            "REFUSED: torch_npu is loaded in this process. This check is CPU-only and"
            " must never touch the device stack."
        )


def main() -> int:
    _refuse_device_stack()
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