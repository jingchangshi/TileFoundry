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


def install_entry(module, rewritten_entry, *, conformance=None, tl_artifact=None):
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
    # Native TIR verification alone cannot authorize publication, but neither may
    # this stage refuse unconditionally: a permanent refusal is indistinguishable
    # from every other one, so it cannot be satisfied, narrowed, or verified - the
    # same defect this tree has now removed four times (R7). The TL layer is still
    # REQUIRED; it is simply judged from real objects instead of declared absent.
    #
    # Four conditions, each named separately so a caller learns WHICH one is short
    # instead of getting one blanket code:
    #   (i)   the verifier receipt, already required above;
    #   (ii)  a TL conformance verdict the READER recomputed to VERIFIED;
    #   (iii) compiled ABI agreement, when a binary was produced;
    #   (iv)  candidate object identity, checked above by object identity.
    # The artifact comes in as an ARGUMENT: NativeVerdict (plan/verify.py) does not
    # carry one, and adding a field there is outside this stage's write scope. A
    # caller that actually emitted and read a TL program passes the REAL artifact it
    # verified, so the layer is judged from that object rather than from a claim.
    conformance_artifact = tl_artifact if tl_artifact is not None else getattr(
        conformance, "tl_artifact", None)
    if conformance_artifact is None:
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "the verifier receipt carries no TL artifact, so the TL layer cannot be "
            "judged for this candidate")
    from ..codegen.tl_reader import read_actual_tl

    parsed = getattr(conformance_artifact, "parsed_prim_func", None)
    if parsed is None:
        raise PlanError("INSTALL_CANDIDATE_UNVERIFIED",
                        "the TL artifact carries no parsed PrimFunc to read")
    # The artifact must describe THIS candidate. The reader checks that the identity
    # is non-empty and that the SOURCE TEXT matches its own digests, but it cannot
    # know which TIR the artifact was supposed to come from - measured: replacing
    # tir_identity with a fabricated value still reads VERIFIED, because nothing
    # downstream compares it with the entry. Comparing it here is what makes the
    # artifact-to-candidate binding real rather than assumed.
    expected_tir = function_identity(rewritten_entry)
    if getattr(conformance_artifact, "tir_identity", "") != expected_tir:
        raise PlanError(
            "ARTIFACT_TIR_IDENTITY_MISMATCH",
            "the TL artifact was emitted from a different TIR than this candidate: "
            "the artifact states %r while the candidate hashes to %r"
            % (getattr(conformance_artifact, "tir_identity", ""), expected_tir))
    verdict = read_actual_tl(conformance_artifact, parsed)
    # The reader owns its code names; this stage reuses them rather than renaming a
    # failure that belongs to the emission.
    for code, _expected, _actual in verdict.refusals:
        raise PlanError(code,
                        "the TL reader refused this candidate; a refused emission "
                        "never authorizes installation")
    for code, _expected, _actual in verdict.gaps:
        raise PlanError(code,
                        "a required layer was never observed for this candidate")
    if verdict.status != "VERIFIED":
        raise PlanError("INSTALL_CANDIDATE_UNVERIFIED",
                        "the recomputed TL verdict is %s, not VERIFIED"
                        % (verdict.status,))
    # ABI, only when this artifact went through the compile site. With no binary the
    # layer is absent rather than passing, and the export gate reports that
    # separately - so it is not asserted here.
    compiled = getattr(conformance_artifact, "compiled", None)
    if compiled is not None:
        from ..codegen.ascend_binary import CompileRefusal, check_compiled_abi

        parameter_count = len(tuple(getattr(rewritten_entry, "params", ()) or ()))
        output_count = int(getattr(rewritten_entry, "output_count", 0) or 0)
        try:
            check_compiled_abi(compiled, parameter_count, output_count)
        except CompileRefusal as refusal:
            raise PlanError(refusal.code,
                            "the compiled binary disagrees with the ABI this candidate "
                            "declares: %s" % (refusal.detail,)) from refusal
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

    # ---- apply / verify -------------------------------------------------------
    # These were MISSING, and their absence is why the production entry could never
    # report ready=True while a hand-assembled test could: the transaction stopped
    # after storage, so applied never equalled SINGLE_TRANSACTION. The stages below
    # are the real ones, in the frozen order - nothing here is a placeholder.
    from .mainline import apply_symbolic_pipeline as _apply
    from .mainline import verify_actual_program as _verify

    try:
        candidate, source_map = _apply(entry, plan, completion, storage)
        stages.append("apply")
    except Exception as error:  # noqa: BLE001
        return _refused_record(module, stages, "apply", error)
    candidate_module = candidate if isinstance(candidate, Module) else None
    if candidate_module is None:
        return _refused_record(
            module, stages, "apply",
            PlanError("MAINLINE_CANDIDATE_NOT_MODULE",
                      "apply_symbolic_pipeline returned %s, not a candidate Module"
                      % (type(candidate).__name__,)))
    try:
        verdict = _verify(module, candidate_module, plan, completion, storage,
                          source_map=source_map)
        stages.append("verify")
    except Exception as error:  # noqa: BLE001
        return _refused_record(module, stages, "verify", error)
    if getattr(verdict, "status", "") != "VERIFIED":
        return _refused_record(
            module, stages, "verify",
            PlanError(getattr(verdict, "code", "") or "MAINLINE_VERIFY_REFUSED",
                      str(getattr(verdict, "detail", ""))[:300]))

    # ---- emit / compile -------------------------------------------------------
    # Emit from the CANDIDATE: the artifact must describe the TIR that will be
    # installed, and the candidate is a distinct object from the source entry.
    from ..codegen import compile_supported_artifact, emit_supported_tl

    try:
        artifact = emit_supported_tl(candidate_module, plan, completion, storage,
                                     candidate_module.entry_function().target)
    except Exception as error:  # noqa: BLE001
        return _refused_record(module, stages, "apply", error)
    try:
        artifact = compile_supported_artifact(artifact)
    except Exception as error:  # noqa: BLE001 - the compile site states why
        # A blocked toolchain is not a stage failure of the transaction: the source
        # layer succeeded, and the gate reports the binary layer separately. Record
        # what ran and let the gate name the binary gap.
        record = _refused_record(module, stages, "verify", error)
        _attach_artifacts(module, plan, completion, storage, artifact, record)
        return record

    # ---- install --------------------------------------------------------------
    try:
        installed = install_entry(module, candidate_module.entry_function(),
                                  conformance=verdict, tl_artifact=artifact)
        stages.append("install")
    except Exception as error:  # noqa: BLE001
        record = _refused_record(module, stages, "install", error)
        _attach_artifacts(module, plan, completion, storage, artifact, record)
        return record

    # The record describes the CALLER module, because that is what the gate is asked
    # about: install_entry returns a NEW Module and leaves the source untouched, so
    # taking identity from the installed one made ir_identity and entry_identity
    # disagree with the module under judgement (measured: STALE_IR_IDENTITY plus
    # ENTRY_IDENTITY_MISMATCH). The candidate entry is recorded separately - it is
    # the object the artifact binds to.
    record = _applied_record(module, module.entry_function(), stages, plan,
                             completion, storage, artifact=artifact,
                             candidate_entry=installed.entry_function())
    # 'export' is the gate's own judgement, made after this record exists; it is
    # recorded by require_export_ready rather than claimed here.
    # install_entry returns a NEW Module whose metadata is a COPY, so writing only to
    # the installed object leaves the CALLER module with none of this evidence -
    # measured: the gate then reported every stage unevidenced because the caller
    # metadata was empty. The artifacts go to BOTH, so the two agree.
    _attach_artifacts(installed, plan, completion, storage, artifact, record)
    _attach_artifacts(module, plan, completion, storage, artifact, record)
    return record


def _attach_artifacts(module, plan, completion, storage, artifact, record):
    """Put the real artifacts where the export gate reads them.

    ONE place, so a stage that stops early still leaves the evidence it DID produce -
    which is what lets the gate name the layer that is missing instead of reporting
    every layer absent at once.
    """
    from ..passes.gate_layers import CANDIDATE_ENTRY_KEY
    from ..passes.p3_chain import (CHAIN_KEY, COMPLETION_KEY, PLAN_KEY, STORAGE_KEY,
                                   TL_ARTIFACT_KEY)

    # The candidate ENTRY object, so the gate can bind the artifact to the program
    # that will actually run - an identity string would be another self-report.
    entry = None
    try:
        entry = module.entry_function()
    except Exception:  # noqa: BLE001 - an unidentifiable module is the gate's gap
        entry = None
    # The CANDIDATE entry is written by _applied_record (or by the install stage).
    # Writing the SOURCE entry here would point the artifact binding at the wrong
    # object and make a correct artifact look misattributed.
    if CANDIDATE_ENTRY_KEY not in module.metadata and entry is not None:
        module.metadata[CANDIDATE_ENTRY_KEY] = entry
    module.metadata[PLAN_KEY] = plan
    module.metadata[COMPLETION_KEY] = completion
    module.metadata[STORAGE_KEY] = storage
    module.metadata[TL_ARTIFACT_KEY] = artifact
    module.metadata[CHAIN_KEY] = record


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


def _applied_record(module, entry, stages, plan, completion, storage, *, artifact,
                    candidate_entry=None):
    """APPLIED, recording the stages that REALLY ran, in frozen order.

    The artifact is passed in rather than produced here: by the time this runs, apply,
    verify, compile and install have happened in _run_transaction, and re-emitting
    would produce a second, possibly different artifact. The applied tuple is the
    stages actually completed - never SINGLE_TRANSACTION by assumption, which is the
    self-report the export gate must not (and no longer does) trust:
    gate_layers.chain_gaps re-derives completion from the artifacts instead.
    """
    from ..passes.gate_layers import CANDIDATE_ENTRY_KEY
    from ..passes.p3_chain import ChainRecord, ir_identity

    if candidate_entry is not None:
        module.metadata[CANDIDATE_ENTRY_KEY] = candidate_entry
    return ChainRecord(
        ir_identity=ir_identity(module),
        entry_identity=function_identity(entry),
        plan_identity=plan.plan_identity,
        completion_identity=completion.identity,
        storage_identity=storage.identity,
        tl_identity=artifact.tl_identity,
        pipeline_status="APPLIED",
        applied=tuple(stages),
        stopped_at="",
        reason="",
    )
