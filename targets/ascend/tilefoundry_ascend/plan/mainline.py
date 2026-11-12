"""Frozen native mainline transaction surface.

This module is the one place where the compiler turns a checked TIR function
into a rewritten symbolic entry. It has no compatibility schema: the transaction
is fixed by INTERFACE_FREEZE_NATIVE_MAINLINE.md and SYMBOLIC_APPLY_SPEC.md.

The three implemented stages:
    apply_symbolic_pipeline -> private candidate Module + SemanticSourceMap
    verify_actual_program   -> NativeVerdict bound to the REAL objects
    install_entry           -> new Module whose entry is that exact candidate
compile_mainline (the whole transaction) remains explicitly NOT IMPLEMENTED: it
must not report APPLIED while emit/read_actual_tl is still missing.
"""
from __future__ import annotations

from tilefoundry.ir.core.module import Module

from .identity import function_identity

__all__ = [
    "MainlineError",
    "MainlineNotImplemented",
    "PlanError",
    "SINGLE_TRANSACTION",
    "apply_symbolic_pipeline",
    "compile_mainline",
    "install_entry",
    "verify_actual_program",
]

SINGLE_TRANSACTION = (
    "freeze", "analyze", "solve", "resolve", "completion",
    "storage", "apply", "verify", "install", "export",
)


class PlanError(ValueError):
    """A malformed plan or failed installation, with a stable named code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


class MainlineError(RuntimeError):
    """A named refusal from the single native transaction."""


class MainlineNotImplemented(MainlineError):
    """The real transaction stage is not implemented yet; never report APPLIED."""


def apply_symbolic_pipeline(entry, plan, completion, storage, *, original=None):
    """Materialize the plan into a PRIVATE candidate module + SemanticSourceMap."""
    from .apply import apply_symbolic_pipeline as _apply

    return _apply(entry, plan, completion, storage, original=original)


def verify_actual_program(original, rewritten, plan, completion, storage, *, source_map=None):
    """Prove the private candidate corresponds to the frozen source; emit a verdict."""
    from .verify import verify_actual_program as _verify

    return _verify(original, rewritten, plan, completion, storage, source_map=source_map)


def install_entry(module, rewritten_entry, *, conformance=None):
    """Install the VERIFIED candidate entry; any failure keeps the original module.

    conformance must be the NativeVerdict produced by verify_actual_program: it
    binds the original module, the candidate module and the candidate entry as
    REAL objects. A dict, a self-made object, a same-name fresh function or a
    same-content hash is refused. The original Module is never mutated; a new
    Module with exactly the entry position replaced is returned.
    """
    from .verify import (
        VERDICT_VERIFIED,
        NativeVerdict,
        _registered_receipt,
    )

    entry_name = getattr(module, "entry", None)
    if not entry_name:
        raise PlanError("PLAN_ENTRY_UNKNOWN", "module has no entry name")
    functions = tuple(getattr(module, "functions", ()) or ())
    if not any(getattr(function, "name", None) == entry_name for function in functions):
        raise PlanError("PLAN_ENTRY_UNKNOWN", "module entry %r is not a function" % (entry_name,))
    if not isinstance(module, Module):
        raise PlanError("INSTALL_MODULE_TYPE", "install_entry needs the real source Module")
    if conformance is None:
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "install_entry requires the actual verify_actual_program conformance result; "
            "a same-name candidate and caller metadata are not verification",
        )
    if isinstance(conformance, dict) or not isinstance(conformance, NativeVerdict):
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "install_entry refuses a self-reported conformance value; the real NativeVerdict "
            "object from verify_actual_program is required",
        )
    if conformance.status != VERDICT_VERIFIED:
        raise PlanError(
            "INSTALL_NOT_VERIFIED",
            "the verifier refused the candidate: %s: %s"
            % (conformance.code, conformance.detail),
        )
    receipt = getattr(conformance, "receipt", None)
    if receipt is None:
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "the verdict carries no verifier receipt; a hand-made NativeVerdict is not "
            "installable evidence",
        )
    if any(getattr(check, "status", None) != "OK" for check in conformance.checks):
        raise PlanError(
            "INSTALL_CHECKS_NOT_PASSED",
            "the verdict contains a check that did not pass",
        )
    if module is not conformance.original_module_ref:
        raise PlanError(
            "INSTALL_MODULE_MISMATCH",
            "the verdict was produced for a different original module object",
        )
    original_entry = module.entry_function()
    if original_entry is not conformance.original_entry_ref:
        raise PlanError(
            "INSTALL_ORIGINAL_ENTRY_MISMATCH",
            "the module's current entry is not the object the verifier read",
        )
    if rewritten_entry is not conformance.candidate_entry_ref:
        raise PlanError(
            "INSTALL_ENTRY_MISMATCH",
            "the rewritten entry is not the object the verifier proved",
        )
    candidate_module = conformance.candidate_module_ref
    if candidate_module is None or candidate_module.entry_function() is not rewritten_entry:
        raise PlanError(
            "INSTALL_CANDIDATE_MODULE_MISMATCH",
            "the candidate module does not own the verified candidate entry object",
        )
    if function_identity(original_entry) != conformance.original_entry_identity:
        raise PlanError("INSTALL_STALE_ORIGINAL",
                        "the original entry changed after it was verified")
    candidate_identity = function_identity(rewritten_entry)
    if candidate_identity != conformance.candidate_entry_identity:
        raise PlanError("INSTALL_STALE_CANDIDATE",
                        "the candidate entry changed after it was verified")
    if conformance.entry_identity != candidate_identity:
        raise PlanError("INSTALL_ENTRY_IDENTITY_MISMATCH",
                        "the verdict's entry identity is not the candidate's current identity")

    # The installable capability is the verifier's process-local receipt for THIS
    # candidate object. A dataclass field cannot mint one: the registry is only
    # written by verify_actual_program, and the receipt must be that same object.
    registered = _registered_receipt(candidate_module, candidate_identity)
    if registered is None or registered is not receipt:
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "no verifier receipt is registered for this candidate object and identity",
        )
    if receipt.candidate_module_ref is not candidate_module:
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "the receipt was issued for a different candidate module object",
        )
    if receipt.candidate_entry_ref is not rewritten_entry:
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "the receipt was issued for a different candidate entry object",
        )
    if receipt.candidate_identity != candidate_identity:
        raise PlanError("INSTALL_CANDIDATE_UNVERIFIED",
                        "the receipt's candidate identity is stale")
    if not conformance.plan_identity or receipt.plan_identity != conformance.plan_identity:
        raise PlanError("INSTALL_PLAN_IDENTITY_MISMATCH",
                        "the plan identity is missing or was not the verified one")
    source_identity = function_identity(original_entry)
    if (conformance.source_identity != source_identity
            or receipt.source_identity != source_identity):
        raise PlanError("INSTALL_SOURCE_IDENTITY_MISMATCH",
                        "the module being installed into is not the frozen source the "
                        "verifier proved")
    # Native TIR verification alone cannot authorize publication.  The actual TL
    # reader and its conformance verdict do not yet exist in this tree.
    raise PlanError("INSTALL_TL_CONFORMANCE_NOT_IMPLEMENTED",
                    "actual TL conformance must be verified before installation")
    positions = [index for index, function in enumerate(functions) if function is original_entry]
    if len(positions) != 1:
        raise PlanError(
            "INSTALL_ENTRY_POSITION",
            "the verified entry must occupy exactly one function position, found %d"
            % (len(positions),),
        )
    installed = tuple(
        rewritten_entry if function is original_entry else function
        for function in functions
    )
    new_module = Module(
        name=module.name,
        functions=installed,
        entry=module.entry,
        children=dict(getattr(module, "children", {}) or {}),
        target=module.target,
        topologies=getattr(module, "topologies", None),
        metadata=dict(getattr(module, "metadata", {}) or {}),
        methods=dict(getattr(module, "methods", {}) or {}),
    )
    if new_module.entry_function() is not rewritten_entry:
        raise PlanError("INSTALL_ENTRY_NOT_INSTALLED",
                        "the new module's entry is not the verified candidate object")
    return new_module


def compile_mainline(module, authored=None, *, target=None):
    """Run the whole transaction on REAL objects; never report APPLIED on a guess.

    This used to raise MAINLINE_CHAIN_NOT_IMPLEMENTED unconditionally, which made
    every downstream stage unreachable and left the export gate permanently closed.
    It now walks the frozen stages in order and stops at the FIRST one that
    refuses, recording which stage and why - so an APPLIED record is evidence that
    every stage actually ran, and a REFUSED one names where it stopped instead of
    saying "not implemented".

    A stage that cannot be satisfied is NOT skipped and NOT faked: the record comes
    back REFUSED with stopped_at set to that stage. Which stage refuses today is
    measured, not assumed - see tests_native_mainline/test_s0_gate_positive.py.
    """
    if not isinstance(module, Module):
        raise PlanError("MAINLINE_MODULE_TYPE",
                        "compile_mainline needs the real source Module, got %s"
                        % (type(module).__name__,))
    from .numeric import to_schedule_problem
    from .problem import analyze_native_problem
    from .resolved import resolve_schedule
    from .units import build_native_unit_tree
    from ..solver import solve_region

    entry = module.entry_function()
    resolved_target = target if target is not None else entry.target
    stages = []
    try:
        tree = build_native_unit_tree(module, authored=authored)
        problem = analyze_native_problem(tree, None, None, resolved_target)
        stages.append("analyze")
        plan = resolve_schedule(problem, {
            region.region_ref.region_id: solve_region(to_schedule_problem(region))
            for region in problem.regions})
        stages.append("resolve")
    except Exception as error:  # noqa: BLE001 - a refusal IS the record here
        return _refused_record(module, stages, "resolve", error)

    from ..completion import build_symbolic_completion, verify_completion

    effects = tuple(effect for region in problem.regions
                    for effect in region.effects)
    try:
        completion = verify_completion(
            plan, build_symbolic_completion(plan, resolved_target, effects),
            resolved_target)
        stages.append("completion")
    except Exception as error:  # noqa: BLE001
        return _refused_record(module, stages, "completion", error)

    from ..storage import place_resolved_versions

    try:
        storage = place_resolved_versions(plan, completion, resolved_target, None)
        stages.append("storage")
    except Exception as error:  # noqa: BLE001
        return _refused_record(module, stages, "storage", error)

    if getattr(completion, "verification", None) is None:
        # Construction is not certification: without a verified completion there
        # is nothing for the later stages to bind, so this stops here by name.
        return _refused_record(
            module, stages, "completion",
            PlanError("MAINLINE_COMPLETION_UNVERIFIED",
                      "the completion carries no verification receipt"))
    return _applied_record(module, entry, stages, plan, completion, storage)


def _refused_record(module, stages, stopped_at, error):
    """One REFUSED record naming the stage that stopped the transaction.

    entry_identity is recorded even when a LATER stage refuses, because the entry
    the transaction ran against is a fact about this run, not about how far it got.
    Leaving it empty made the export gate report ("entry", "ENTRY_IDENTITY_MISMATCH")
    for a program whose entry was never in question - a gap that named the wrong
    layer, which is exactly the kind of misleading entry this gate exists to avoid.
    """
    from ..passes.p3_chain import ChainRecord, ir_identity

    code = getattr(error, "code", None) or type(error).__name__
    entry = module.entry_function() if hasattr(module, "entry_function") else None
    return ChainRecord(
        ir_identity=ir_identity(module),
        entry_identity=function_identity(entry) if entry is not None else "",
        pipeline_status="REFUSED",
        applied=tuple(stages),
        stopped_at=stopped_at,
        reason=code,
    )


def _applied_record(module, entry, stages, plan, completion, storage):
    """APPLIED, with every artifact attached so the gate can re-derive its own view.

    The artifacts go into module.metadata under the keys the export gate reads, in
    ONE place: the gate never trusts the record's identity strings when it can
    recompute them, but it needs the OBJECTS to recompute from.
    """
    from ..codegen import compile_supported_artifact, emit_supported_tl
    from ..passes.p3_chain import (CHAIN_KEY, COMPLETION_KEY, PLAN_KEY,
                                   STORAGE_KEY, TL_ARTIFACT_KEY, ChainRecord,
                                   ir_identity)

    target = entry.target
    try:
        artifact = emit_supported_tl(module, plan, completion, storage, target)
        stages = tuple(stages) + ("apply",)
    except Exception as error:  # noqa: BLE001
        return _refused_record(module, stages, "apply", error)
    try:
        artifact = compile_supported_artifact(artifact)
        stages = stages + ("verify",)
    except Exception as error:  # noqa: BLE001 - the compile site states why
        # The artifact is still real and still worth attaching: the gate reports
        # the binary layer separately, and a missing compile must not erase the
        # evidence that emit/parse succeeded.
        record = ChainRecord(
            ir_identity=ir_identity(module),
            entry_identity=function_identity(entry),
            plan_identity=plan.plan_identity,
            completion_identity=completion.identity,
            storage_identity=storage.identity,
            tl_identity=artifact.tl_identity,
            pipeline_status="REFUSED",
            applied=stages,
            stopped_at="verify",
            reason=getattr(error, "code", None) or type(error).__name__,
        )
        module.metadata[PLAN_KEY] = plan
        module.metadata[COMPLETION_KEY] = completion
        module.metadata[STORAGE_KEY] = storage
        module.metadata[TL_ARTIFACT_KEY] = artifact
        module.metadata[CHAIN_KEY] = record
        return record

    record = ChainRecord(
        ir_identity=ir_identity(module),
        entry_identity=function_identity(entry),
        plan_identity=plan.plan_identity,
        completion_identity=completion.identity,
        storage_identity=storage.identity,
        tl_identity=artifact.tl_identity,
        pipeline_status="APPLIED",
        applied=stages,
        stopped_at="",
        reason="",
    )
    module.metadata[PLAN_KEY] = plan
    module.metadata[COMPLETION_KEY] = completion
    module.metadata[STORAGE_KEY] = storage
    module.metadata[TL_ARTIFACT_KEY] = artifact
    module.metadata[CHAIN_KEY] = record
    return record
