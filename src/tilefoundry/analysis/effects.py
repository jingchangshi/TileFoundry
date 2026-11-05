"""Effect roots for dead-code elimination over value-form HIR.

D0 HIR is pure SSA: a loop's observable effect is exactly the value it produces.
So the effect-aware rule is a reachability rule with one explicit exclusion:

1. EXTERNAL WRITE (write GM / return value / loop carry) -- a loop whose produced
   value is reachable from the function result is retained, because that value IS
   the externally visible write. "Returned" and "loop-carried into a returned value"
   are the same statement in SSA: reachable from Function.body.
2. PRIVATE SMEM DEAD VALUE -- a loop whose produced value has no consumer is not
   reachable and may be deleted. Deleting it changes nothing anyone can observe.
3. INSTRUCTION OPERAND WRITE IS NOT AN HIR MUTATION -- an instruction (tf.schedule)
   declares its operand's memory effect, but writing a register/UB window does NOT
   turn the old Var into a new value. It is therefore NOT a DCE root on its own;
   only a consumed produced value keeps a loop. Counting operand writes as effects is
   exactly the "keep every store loop" over-retention this module forbids.

RETENTION ASSUMPTION (explicit, D0 value-form SSA)
-----------------------------------------------------
This module ONLY implements value-form retention. It assumes every externally visible
write is carried by a PRODUCED VALUE that is reachable from the function result. D0 does
NOT model imperative mutation of an existing Var: an instruction that writes a GM operand
is NOT treated as producing a new externally visible value. A program whose only external
effect is such an unconsumed operand write cannot be represented, and
assert_value_form_retention refuses it by name (DCE_VALUE_FORM_ASSUMPTION_VIOLATED) rather
than silently dropping it or pretending it was carried. If D0 ever grows
imperative-mutation semantics, this assumption must be revisited and this module widened
deliberately.

WIRING (production vs test-only), per function:
  * reachable_loops                -- PRODUCTION: parser/pipeline_intents uses it to decide
                                      which authored pipeline declarations survived.
  * reachable_loop_ids, loop_is_retained -- helper/test-only (not on a production path).
  * storage_of, is_external_storage, value_is_external -- test-only predicates.
  * instruction_operand_writes     -- test-only; documents that an operand WRITE is not a
                                      DCE root.
  * classify_dropped_loop          -- test-only; classifies a dropped loop.
  * assert_value_form_retention    -- PRODUCTION-callable guard (see assumption above).
"""

from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.pipeline import PipelineError
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import collect_exprs

__all__ = [
    "storage_of",
    "is_external_storage",
    "value_is_external",
    "instruction_operand_writes",
    "reachable_loop_ids",
    "reachable_loops",
    "loop_is_retained",
    "classify_dropped_loop",
    "assert_value_form_retention",
]


def storage_of(value: object) -> StorageKind | None:
    """The storage class of a tensor-typed value, or None for a non-tensor."""
    type_ = getattr(value, "type", None)
    return type_.storage if isinstance(type_, TensorType) else None


def is_external_storage(storage: object) -> bool:
    """Whether storage is externally visible (device/global memory)."""
    return storage is StorageKind.GMEM


def value_is_external(value: object) -> bool:
    """Whether the value's own type is externally visible storage (GMEM).

    Wiring: TEST-ONLY predicate (used by DCE controls, not on a production path).
    """
    return is_external_storage(storage_of(value))


def _op_schema(op: object):
    return getattr(type(op), "_op_schema", None)


def instruction_operand_writes(call: object) -> tuple[tuple[int, StorageKind | None], ...]:
    """(arg index, storage) for every operand the instruction declares WRITE.

    Wiring: TEST-ONLY; documents that an operand WRITE is NOT a DCE root.


    This is INSTRUCTION operand effect only. It is reported for diagnosis and is
    deliberately NOT consulted by loop_is_retained: an operand write to a register or
    a UB window is not an HIR mutation of the old Var, so it must never be what keeps a
    loop alive. Only a consumed produced value is a DCE root.
    """
    if not isinstance(call, Call):
        return ()
    schema = _op_schema(call.target)
    signature = getattr(schema, "signature", None)
    if not signature:
        return ()
    params = tuple(param for param in signature if getattr(param, "kind", None) == "input")
    writes: list[tuple[int, StorageKind | None]] = []
    for index, argument in enumerate(call.args):
        if index >= len(params):
            break
        effect = getattr(params[index], "effect", None)
        if isinstance(effect, MemoryEffect) and (effect & MemoryEffect.WRITE):
            writes.append((index, storage_of(argument)))
    return tuple(writes)


def reachable_loop_ids(function) -> frozenset[int]:
    """Ids of every LoopRegion reachable from the function's returned value."""
    body = getattr(function, "body", None)
    if body is None:
        return frozenset()
    return frozenset(id(expr) for expr in collect_exprs(body) if isinstance(expr, LoopRegion))


def reachable_loops(function) -> tuple[LoopRegion, ...]:
    """Every LoopRegion reachable from the function's returned value, in body order.

    Wiring: PRODUCTION (parser/pipeline_intents uses this to decide which authored
    pipeline declarations survived DCE).
    """
    body = getattr(function, "body", None)
    if body is None:
        return ()
    return tuple(expr for expr in collect_exprs(body) if isinstance(expr, LoopRegion))


def loop_is_retained(loop: LoopRegion, reachable_ids: frozenset[int]) -> bool:
    """A loop is retained iff its produced value is reachable from the result.

    Wiring: TEST-ONLY helper.
    """
    return id(loop) in reachable_ids


def classify_dropped_loop(loop: LoopRegion) -> str:
    """How a NOT-reachable loop's disappearance must be read.

    Wiring: TEST-ONLY classifier.


    * pipeline -- an authored pipeline declaration; its loss is refused by name
      (PIPELINE_DECLARATION_DROPPED_BY_DCE), never silently accepted.
    * private-dead -- a produced value with no consumer; deleting it is allowed.
    """
    if getattr(loop, "pipeline", None) is not None:
        return "pipeline"
    return "private-dead"


def assert_value_form_retention(function, candidate_loops=()) -> dict:
    """Assert the value-form SSA retention assumption for dropped loops, or refuse by name.

    D0 cannot express an unconsumed external (GMEM) operand WRITE as a new value. A dropped
    loop (one not reachable from the result) whose body contains an instruction writing a
    GMEM operand is therefore exactly the case the value-form model cannot represent: it is
    refused with DCE_VALUE_FORM_ASSUMPTION_VIOLATED instead of being silently deleted. A
    dropped loop that only writes private storages (SMEM/RMEM/CARRIER) is a private dead
    value and is allowed to disappear.

    Wiring: PRODUCTION-callable guard. Callers pass the LoopRegions they built (including
    ones DCE may drop); pass no candidates when only the surviving body is available.
    """
    reachable = reachable_loop_ids(function)
    inspected = 0
    dropped = 0
    for loop in candidate_loops:
        if id(loop) in reachable:
            continue
        dropped += 1
        for expr in collect_exprs(loop.body):
            if not isinstance(expr, Call):
                continue
            for index, storage in instruction_operand_writes(expr):
                inspected += 1
                if is_external_storage(storage):
                    raise PipelineError(
                        "DCE_VALUE_FORM_ASSUMPTION_VIOLATED",
                        "dropped loop writes a GMEM operand (arg %d of %r) whose produced "
                        "value is not consumed; D0 value-form SSA cannot represent an "
                        "unconsumed external write" % (index, type(expr.target).__name__),
                    )
    return {
        "value_form": "ssa-reachability",
        "candidates": len(tuple(candidate_loops)),
        "dropped": dropped,
        "operand_writes_inspected": inspected,
    }
