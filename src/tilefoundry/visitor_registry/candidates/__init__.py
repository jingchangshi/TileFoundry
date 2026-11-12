"""Registered HIR-to-TIR candidate pairings."""

from __future__ import annotations

import importlib
import pkgutil

from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.core.execution import region_requirement
from tilefoundry.ir.core.op import Op, op_identifier
from tilefoundry.ir.core.param_def import MemoryEffect, collect_param_defs
from tilefoundry.ir.hir.schedule import ScheduleRefusal

_CANDIDATES: dict[type, list[type]] = {}
_CALLBACKS: dict[type, object] = {}

#: Target-scoped candidate instructions, keyed by (target name, HIR op type).
#:
#: These are NOT global candidates: they are only offered when that target is the one
#: lowering the program. Keeping them out of _CANDIDATES is deliberate -- appending an
#: Ascend op to the global HIR Binary/Copy list would make sole_candidate return None
#: for every target (including CUDA), and the default has to stay byte-identical.
_TARGET_CANDIDATES: dict[tuple[str, type], list[type]] = {}


def register_candidates(hir_op: type, tir_ops: tuple[type, ...]):
    """Append instructions; a decorated callback supplies attributes and operand types."""
    registered = _CANDIDATES.setdefault(hir_op, [])
    for tir_op in tir_ops:
        if tir_op not in registered:
            registered.append(tir_op)

    def decorate(callback):
        if hir_op in _CALLBACKS:
            raise ValueError(f"{hir_op.__name__} already has a candidate operand callback")
        _CALLBACKS[hir_op] = callback
        return callback

    return decorate


def candidate_ops(hir_op: type) -> tuple[type, ...]:
    """Return every registered TIR carrier for one HIR operation."""
    return tuple(_CANDIDATES.get(hir_op, ()))


def _instruction_attributes(hir_op: Op, op_type: type[Op]) -> dict:
    """Copy same-named attributes and reject information the instruction cannot express."""
    parameters = collect_param_defs(op_type)
    names = {param.name for param in parameters}
    for param in collect_param_defs(type(hir_op)):
        if param.kind != "attribute" or param.name in names:
            continue
        value = getattr(hir_op, param.name)
        if not param.has_default or value != param.default:
            raise ScheduleRefusal(
                f"{type(hir_op).__name__}.{param.name}={value!r} has no counterpart on {op_type.__name__}"
            )
    return {
        param.name: getattr(hir_op, param.name)
        for param in parameters
        if param.kind == "attribute" and hasattr(hir_op, param.name)
    }


def instruction_from_hir(hir_op: Op, op_type: type[Op]) -> Op | None:
    """Build an instruction from same-named HIR attributes, or None if incomplete."""
    attributes = _instruction_attributes(hir_op, op_type)
    if any(
        param.kind == "attribute" and not param.has_default and param.name not in attributes
        for param in collect_param_defs(op_type)
    ):
        return None
    return op_type(**attributes)


def candidate_operands(call, op_type, *, target=None):
    """The registered callback, or the default instruction-order operand mapping.

    ``target`` is threaded through for consistency with the rest of target-aware
    selection; the callback contract is currently target-independent, so it is not
    consulted here, but keeping it in the signature means a caller never has to know
    which side of the selection consumes it.
    """
    callback = _CALLBACKS.get(type(call.target))
    if callback is not None:
        return callback(call, op_type)
    attributes = _instruction_attributes(call.target, op_type)
    reads = iter(arg.type for arg in call.args)
    operands = tuple(
        call.type if param.effect & MemoryEffect.WRITE else next(reads)
        for param in collect_param_defs(op_type)
        if param.kind == "input" and not param.optional
    )
    return attributes, operands


def sole_candidate(hir_op: Op, *, target=None, inside: bool = False, active_kind=None) -> Op | None:
    """The single registered instruction, or None when the choice is not the default.

    With ``target=None`` this is the historical, GLOBAL query: exactly one globally
    registered carrier for the HIR op, resolved from same-named attributes. That is the
    behaviour every existing caller (and CUDA) depends on, so it is unchanged.

    With a ``target`` supplied, target-scoped candidates are offered first: the result is
    the target-resolved instruction for the requested placement, ``None`` if the target
    declares no scoped candidate for this op (fall back to the global query), and a
    ``ScheduleRefusal`` if the scoped set is ambiguous or cannot fit. This is what makes
    selection target-aware in one place instead of a later converter-only substitution.
    """
    if target is not None:
        scoped = select_instruction(hir_op, target=target, inside=inside, active_kind=active_kind)
        if scoped is not None:
            return scoped
    call = hir_op if isinstance(hir_op, Call) else None
    target = call.target if call is not None else hir_op
    candidates = candidate_ops(type(target))
    if len(candidates) != 1:
        return None
    if call is None:
        return instruction_from_hir(target, candidates[0])
    attributes, _types = candidate_operands(call, candidates[0])
    if any(
        param.kind == "attribute" and not param.has_default and param.name not in attributes
        for param in collect_param_defs(candidates[0])
    ):
        return None
    return candidates[0](**attributes)


def _op_name(op_type: type) -> str:
    """Stable name for a candidate even if it is not a registered operation.

    Registered ops use their dialect-qualified identifier; a test-only/one-off class
    (used to prove the ambiguity refusal) has no schema, so its class name is used
    instead of raising. The refusal text must never depend on registration succeeding.
    """
    return op_identifier(op_type) if getattr(op_type, "_op_schema", None) is not None else op_type.__name__


def _target_key(target) -> str | None:
    if target is None:
        return None
    return getattr(target, "name", None) or type(target).__name__


def register_target_candidates(target_key: str, hir_op: type, tir_ops: tuple[type, ...]) -> None:
    """Offer *tir_ops* as candidate instructions for *hir_op* ONLY under *target_key*.

    Distinct from register_candidates on purpose: this never touches the global list, so
    it cannot change what a different target sees or make a global sole_candidate
    ambiguous. A target opts in by naming itself here.
    """
    registered = _TARGET_CANDIDATES.setdefault((target_key, hir_op), [])
    for tir_op in tir_ops:
        if tir_op not in registered:
            registered.append(tir_op)


def target_candidate_ops(target, hir_op: type) -> tuple[type, ...]:
    """Candidate instructions *target* scoped to *hir_op* (empty for none)."""
    key = _target_key(target)
    if key is None:
        return ()
    return tuple(_TARGET_CANDIDATES.get((key, hir_op), ()))


def _declared_operand_storages(op_type: type) -> tuple:
    """The storage each required input operand DECLARES, in signature order."""
    return tuple(
        getattr(getattr(param, "pattern", None), "storage", None)
        for param in collect_param_defs(op_type)
        if param.kind == "input" and not param.optional
    )


def _operand_storages(operands) -> tuple:
    return tuple(getattr(operand, "storage", None) for operand in operands)


def _storages_fit(declared: tuple, actual: tuple) -> bool:
    """Every operand whose declared storage is stated must actually be in it.

    A declared storage of None is unconstrained and fits anything; a stated storage must
    match exactly. This is what tells ascend_gm_to_ub (gmem->smem) apart from
    ascend_ub_to_gm (smem->gmem) for the SAME generic reshard, and keeps a direction-less
    op from being chosen for the wrong direction.
    """
    if len(declared) != len(actual):
        return False
    return all(want is None or got is want for want, got in zip(declared, actual))


def select_instruction(hir_op, *, target, inside: bool, active_kind=None):
    """Target-aware automatic instruction for one HIR op, or None if target declares none.

    Resolution, in one place shared by lowering and reporting:
    * ask only the TARGET-scoped candidates (global list untouched);
    * keep those whose DECLARED placement matches where the op sits -- "inside" for a
      mode-scoped generic op, "outside" otherwise -- and, inside, whose declared kind is
      the active mode;
    * among those, keep those whose declared operand storages fit the actual operands;
    * exactly one -> select it; more than one -> refuse as ambiguous; none -> refuse with
      the candidates it did have.

    Returns None only when the target declares NO scoped candidate for this HIR op, so the
    caller keeps the existing global selection (this is what leaves CUDA byte-identical).
    """
    call = hir_op if isinstance(hir_op, Call) else None
    target_op = call.target if call is not None else hir_op
    hir_type = type(target_op)
    scoped = target_candidate_ops(target, hir_type)
    if not scoped:
        return None
    want = "inside" if inside else "outside"
    fitting = []
    for op_type in scoped:
        requirement = region_requirement(op_type)
        if requirement is None:
            continue
        if requirement.placement != want:
            continue
        if inside and active_kind is not None and requirement.kind != active_kind:
            continue
        if call is not None:
            attributes, operands = candidate_operands(call, op_type, target=target)
            if not _storages_fit(_declared_operand_storages(op_type), _operand_storages(operands)):
                continue
        fitting.append(op_type)
    listed = ", ".join(_op_name(op_type) for op_type in scoped)
    if len(fitting) == 1:
        op_type = fitting[0]
        if call is None:
            return None
        attributes, _types = candidate_operands(call, op_type, target=target)
        return op_type(**attributes)
    if len(fitting) > 1:
        chosen = ", ".join(_op_name(op_type) for op_type in fitting)
        raise ScheduleRefusal(
            f"{hir_type.__name__}: ambiguous automatic instruction selection among "
            f"[{chosen}] for {want} placement; write tf.schedule to choose the instruction"
        )
    raise ScheduleRefusal(
        f"{hir_type.__name__}: none of the declared instructions [{listed}] fits {want} "
        "placement and these operand storages; write tf.schedule to choose the instruction"
    )

def _auto_import(pkg_name: str) -> None:
    package = importlib.import_module(pkg_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, f"{pkg_name}."):
        importlib.import_module(module_name)


_auto_import(__name__)

__all__ = [
    "candidate_operands",
    "instruction_from_hir",
    "sole_candidate",
    "candidate_ops",
    "register_candidates",
    "register_target_candidates",
    "target_candidate_ops",
    "select_instruction",
]
