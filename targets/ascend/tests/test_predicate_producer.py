"""M3: a predicate operand must be PRODUCED by an instruction in its region.

Direct-TIR cases. The property under test is data flow, not shape: a mask is a bool
register, so an allocated-but-never-written mask type-checks, matches the bool-in-rmem
operand pattern, passes the lane rule (bool exempt) and passes the dtype rule (bool is
a mask, not an element). Dropping pset therefore left a program whose every operand is
well formed and whose mask nothing ever wrote -- and it was ACCEPTED. This file pins the
rule that closes that hole, and the positive control that proves the rule does not just
refuse the whole shape.

A refusal is only evidence for this contract when it comes from the layer that STATES
it, so the negative asserts the predicate-producer message and records the layer name
next to it rather than treating any exception as a pass.

TWO NEGATIVES, because the rule is about ORDER, not just presence:
  * mask_without_producer_is_refused -- no pset at all
  * late_pset_is_refused            -- pset placed AFTER the vadd that reads the mask
The second is the one a presence-only check misses: the resolved TIR is a Sequential,
so a writer that runs later has not run when the reader reads. The producer must DOMINATE
the consumer (precede it in statement order), and the swapped-order case must FAIL at the
same predicate-producer layer.

WHAT THE FIRST RUN MUST SHOW: the positive control ACCEPTED (or every negative passes
for free), and BOTH negatives refused with the predicate-producer text. Exit codes:
0 PASS, 1 FAIL, 2 ERROR.

HARNESS STATUS -- a DOCUMENTED ADDITIONAL OBLIGATION, not part of the default run:
these three cases are NOT wired into targets/ascend/tests/run_acceptance.sh, because that
script is sha256-pinned into the current receipt and editing it would invalidate the pin.
Until a future default-harness revision adds this file, they must be invoked directly
(pytest is not installed in the conda env; use the __main__ runner below) and their
result recorded separately. They are green today (3/3) and are an obligation precisely
because nothing in the default path would catch a regression here yet.
"""
from __future__ import annotations

import sys

import tilefoundry_ascend  # noqa: F401  (registers SimdVF and its instructions)

from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import DType, Mesh, TensorType, Topology, make_tensor_type
from tilefoundry_ascend.ir.ops import Pset, Vadd, Vld
from tilefoundry.visitor_registry.verify import verify_prim_function

if "torch_npu" in sys.modules:
    raise SystemExit(
        "REFUSED: torch_npu is loaded in this process. This check is CPU-only and must"
        " never touch the device stack."
    )

KIND = "ascend.simdvf"
LANES = 64
AIV = Mesh((Topology("aiv", 1),), (1,), ("core",))


def _tensor(shape=(LANES,), dtype="f32", storage="rmem") -> TensorType:
    """A tensor type with REAL DType descriptors (make_tensor_type does not resolve
    dtype strings; a string dtype would refuse everything and pass every negative)."""
    return make_tensor_type(shape, dtype=DType.from_name(dtype), storage=storage)


def _bind(name, type_):
    """Allocate *type_* and return (LetStmt, Var)."""
    bound = Var(name=name, type=type_)
    statement = LetStmt(
        var=bound,
        value=Call(target=AllocTensor(tensor_type=type_), args=(), type=type_),
    )
    return statement, bound


def _function(params, *body):
    return PrimFunction(
        name="fn",
        params=tuple(params),
        body=MeshScope(
            AIV,
            Var(name="aiv", type=_tensor(storage="smem")),
            Sequential(body=body),
        ),
        output_count=0,
    )


def _chain(pset: str = "before") -> PrimFunction:
    """The predicated vadd with the mask-writing pset placed:

    "before" -- pset precedes vadd (the correct program)
    "after"  -- pset FOLLOWS vadd (the swapped-order defect)
    "none"   -- no pset at all (the dropped-producer defect)
    """
    ub_stmt, ub = _bind("ub", _tensor(storage="smem"))
    left_stmt, left = _bind("left", _tensor())
    right_stmt, right = _bind("right", _tensor())
    acc_stmt, acc = _bind("acc", _tensor())
    mask_stmt, mask = _bind("mask", _tensor(dtype="bool"))
    inner = [
        left_stmt, right_stmt, acc_stmt, mask_stmt,
        Evaluate(callable=Vld(), args=(ub, left)),
        Evaluate(callable=Vld(), args=(ub, right)),
    ]
    pset_stmt = Evaluate(callable=Pset(elem_width=32), args=(mask,))
    if pset == "before":
        inner.append(pset_stmt)
    inner.append(Evaluate(callable=Vadd(), args=(left, right, mask, acc)))
    if pset == "after":
        inner.append(pset_stmt)
    scope = ExecutionScope(kind=KIND, body=Sequential(body=tuple(inner)))
    return _function((), ub_stmt, scope)


def _refusal(fn) -> str:
    try:
        verify_prim_function(fn)
    except VerifyError as error:
        return str(error)
    raise AssertionError("expected a refusal; the program was accepted")


def _assert_predicate_producer_refusal(message, what):
    if "no producing instruction" not in message:
        raise AssertionError(f"{what}: refused, but not by the predicate-producer rule: {message}")
    if "nothing produced" not in message:
        raise AssertionError(f"{what}: predicate-producer rule fired without its full text: {message}")


def check_positive_control_is_accepted() -> None:
    """With pset BEFORE vadd the chain is ACCEPTED. Without this every negative is free."""
    verify_prim_function(_chain(pset="before"))


def check_mask_without_producer_is_refused() -> None:
    """Drop pset: the mask has no producer and the chain must be refused."""
    message = _refusal(_chain(pset="none"))
    _assert_predicate_producer_refusal(message, "dropped pset")
    print("       refused:", message.splitlines()[0][:150])
    print("       layer=predicate-producer")


def check_late_pset_is_refused() -> None:
    """pset placed AFTER the vadd: the writer does not dominate the reader. Must FAIL.

    A presence-only check passes this (a writer exists), which is exactly the hole the
    architect found. The resolved TIR is a Sequential, so a later writer has not run
    when vadd reads the mask.
    """
    message = _refusal(_chain(pset="after"))
    _assert_predicate_producer_refusal(message, "late pset")
    print("       refused:", message.splitlines()[0][:150])
    print("       layer=predicate-producer (dominance)")


CHECKS = [
    ("positive_control_is_accepted", check_positive_control_is_accepted),
    ("mask_without_producer_is_refused", check_mask_without_producer_is_refused),
    ("late_pset_is_refused", check_late_pset_is_refused),
]


def main() -> int:
    failed = 0
    for name, check in CHECKS:
        try:
            check()
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}")
            print(f"     {error}")
        except Exception as error:  # noqa: BLE001 -- any raise is an ERROR, not a verdict
            failed += 1
            print(f"ERROR {name}: {error!r}")
        else:
            print(f"ok   {name}")
    if failed:
        print(f"{failed}/{len(CHECKS)} checks FAILED")
        return 1
    print(f"{len(CHECKS)}/{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
