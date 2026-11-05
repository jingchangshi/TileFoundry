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

This module states those facts as callable predicates so the retention decision is
readable and testable instead of being an accident of how the body was assembled.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.ir.core.param_def import MemoryEffect
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
]


def storage_of(value: object) -> StorageKind | None:
    """The storage class of a tensor-typed value, or None for a non-tensor."""
    type_ = getattr(value, "type", None)
    return type_.storage if isinstance(type_, TensorType) else None


def is_external_storage(storage: object) -> bool:
    """Whether storage is externally visible (device/global memory)."""
    return storage is StorageKind.GMEM


def value_is_external(value: object) -> bool:
    """Whether the value's own type is externally visible storage (GMEM)."""
    return is_external_storage(storage_of(value))


def _op_schema(op: object):
    return getattr(type(op), "_op_schema", None)


def instruction_operand_writes(call: object) -> tuple[tuple[int, StorageKind | None], ...]:
    """(arg index, storage) for every operand the instruction declares WRITE.

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
    """Every LoopRegion reachable from the function's returned value, in body order."""
    body = getattr(function, "body", None)
    if body is None:
        return ()
    return tuple(expr for expr in collect_exprs(body) if isinstance(expr, LoopRegion))


def loop_is_retained(loop: LoopRegion, reachable_ids: frozenset[int]) -> bool:
    """A loop is retained iff its produced value is reachable from the result."""
    return id(loop) in reachable_ids


def classify_dropped_loop(loop: LoopRegion) -> str:
    """How a NOT-reachable loop's disappearance must be read.

    * pipeline -- an authored pipeline declaration; its loss is refused by name
      (PIPELINE_DECLARATION_DROPPED_BY_DCE), never silently accepted.
    * private-dead -- a produced value with no consumer; deleting it is allowed.
    """
    if getattr(loop, "pipeline", None) is not None:
        return "pipeline"
    return "private-dead"
