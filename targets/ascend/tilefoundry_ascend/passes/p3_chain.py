"""The single native Ascend mainline pass.

The export gate is fail-closed: it reads the actual installed entry and the
actual NativeArtifact the TL emitter produced, and status bytes, dicts and
self-reported identity strings are never proof. It no longer ends in a single
unconditional NOT_IMPLEMENTED byte - the layers it requires are enumerated and
judged separately in passes/gate_layers.py, so a refusal names WHICH layer is
short of evidence. Today every real program is still refused, because the
installed-candidate receipt, the compiled binary and the emitted event
statements do not exist yet.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import PassPoint, register_pass

from ..codegen import PASS_CONFIGS, NativeArtifact
from ..plan.identity import function_identity, module_identity
from ..plan.mainline import MainlineNotImplemented, SINGLE_TRANSACTION, compile_mainline
from ..tf_target import Ascend950PRTarget
from . import gate_layers

__all__ = [
    "CHAIN_KEY", "CHAIN_SCHEMA", "EXPORT_GATE_KEY", "PLAN_KEY", "COMPLETION_KEY",
    "STORAGE_KEY", "TL_ARTIFACT_KEY", "TL_CONFORMANCE_KEY", "ChainRecord",
    "ExportRefusal", "ir_identity", "require_export_ready", "RunAscendNativeMainline",
]

CHAIN_KEY = "ascend.native_mainline_chain"
CHAIN_SCHEMA = "native_mainline"
EXPORT_GATE_KEY = "ascend.export_gate"
PLAN_KEY = "ascend.native_mainline_plan"
COMPLETION_KEY = "ascend.native_mainline_completion"
STORAGE_KEY = "ascend.native_mainline_storage"
TL_ARTIFACT_KEY = "ascend.native_mainline_tl"
TL_CONFORMANCE_KEY = "ascend.native_mainline_tl_conformance"


class ExportRefusal(RuntimeError):
    """A named refusal to publish a runnable artifact."""

    def __init__(self, code: str, detail: str, *, missing=(), gate: str = ""):
        self.code = code
        self.detail = detail
        self.missing = tuple(missing)
        self.gate = gate
        super().__init__("%s: %s" % (code, detail))


def _function_identity(function) -> str:
    """Identity of the ACTUAL function, including non-dataclass Op attributes.

    A name/kind digest is not an identity: a Vadds(value=1.0) and a
    Vadds(value=7.0) would collide, as would Vcvt(part=0)/Vcvt(part=1). The
    program content lives in each Op's published attribute parameters and in the
    binder its Var references resolve to, both of which the shared identity walk
    reads (see plan/identity.py).
    """
    return function_identity(function)


def ir_identity(module: Module) -> str:
    """Identity over every actual function of the module."""
    return module_identity(module)


def _entry_prim(module: Module):
    entry_name = getattr(module, "entry", None)
    for function in getattr(module, "functions", ()) or ():
        if isinstance(function, PrimFunction) and function.name == entry_name:
            return function
    return None


def _get(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _artifact_identity(artifact) -> str:
    for name in ("identity", "source_identity", "tl_identity"):
        value = _get(artifact, name, "")
        if value:
            return str(value)
    return ""


@dataclass(frozen=True)
class ChainRecord:
    ir_identity: str
    entry_identity: str = ""
    plan_identity: str = ""
    completion_identity: str = ""
    storage_identity: str = ""
    tl_identity: str = ""
    pipeline_status: str = "REFUSED"
    applied: tuple = ()
    stopped_at: str = "freeze"
    reason: str = "MAINLINE_NOT_IMPLEMENTED"

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _metadata_artifact_gaps(metadata, record):
    """The STORAGE artifact layer, and nothing else.

    SINGLE OWNERSHIP (task-49). This loop used to cover plan, completion and
    storage in one sweep, comparing each artifact's own identity field with the
    ChainRecord's stored string. Two problems, both measured:

      1. It spoke the same sentence as gate_layers.plan_gaps/completion_gaps in
         the same gap list, which is the duplicate-entry defect the lead found
         (two (tl, TL_ARTIFACT_NOT_ACTUAL) rows in minimal_aiv's gate).
      2. Both sides of its comparison were STORED strings. record.plan_identity
         was copied out of plan.plan_identity at pass time, so comparing them
         cannot notice anything; R20 showed a replaced RegionPlan leaves that
         field byte-identical. An identity check that reads only recorded fields
         is not a check.

    So plan and completion moved to gate_layers, where each is RECOMPUTED from
    the live objects (plan/resolved.py's plan_identity_of, and the completion
    verifier's own re-run). Storage stays here as the honest stored-vs-record
    comparison, and its recompute - storage/__init__.py's
    verify_storage_current, which needs the plan, target and optionally the
    completion - is listed as an outstanding gap rather than claimed as done.
    """
    artifact = metadata.get(STORAGE_KEY)
    if artifact is None:
        return [("storage", "STORAGE_ARTIFACT_MISSING")]
    if isinstance(artifact, dict) or not _artifact_identity(artifact):
        return [("storage", "STORAGE_ARTIFACT_NOT_ACTUAL")]
    if not record.storage_identity or record.storage_identity != _artifact_identity(artifact):
        return [("storage", "STORAGE_IDENTITY_MISMATCH")]
    return []


def _native_artifact_gaps(metadata, entry, record):
    """The artifact-to-RECORD binding layer, and nothing else.

    SINGLE OWNERSHIP (task-49). This function used to also test whether an
    artifact exists, whether it is a real NativeArtifact, whether its parsed
    PrimFunc is present, whether the source text is empty, whether PASS_CONFIGS
    match, whether the ABI counts agree, and whether a stored conformance verdict
    says "verified". Those are the artifact's CONTENT and the content layers are
    owned by passes/gate_layers.py. Keeping both meant the same gap was appended
    twice in the same list - measured: (tl, TL_ARTIFACT_NOT_ACTUAL) appearing two
    times in minimal_aiv's gate.gaps - and the tests hid it by comparing sets, so
    a downstream consumer counting or rendering entries saw a phantom second
    gap. Deduplicating the list would have hidden the structure instead of fixing
    it, so the seam was moved: exactly one function may speak for each layer.

    What stays here is what only this function can judge, because only it holds
    the ChainRecord: the artifact must carry a non-empty identity in every field
    it is supposed to have, and the two fields that tie it to THIS pipeline run
    must equal the record's. A check that reads a self-reported field as proof is
    not kept: the conformance verdict is RECOMPUTED by gate_layers rather than
    taken from metadata, which is strictly stronger and cannot be satisfied by a
    hand-written object.
    """
    artifact = metadata.get(TL_ARTIFACT_KEY)
    if not isinstance(artifact, NativeArtifact):
        # Ownership of "there is no real artifact" lies with gate_layers, which
        # reports it once. Here the layer simply has nothing to bind.
        return []
    gaps = []
    for field in ("source_identity", "plan_identity", "tir_identity", "tl_identity",
                  "target_identity", "toolchain_identity"):
        if not getattr(artifact, field, ""):
            gaps.append(("tl", "ARTIFACT_%s_MISSING" % field.upper()))
    if getattr(artifact, "plan_identity", "") != record.plan_identity:
        gaps.append(("tl", "ARTIFACT_PLAN_IDENTITY_MISMATCH"))
    if getattr(artifact, "tl_identity", "") != record.tl_identity:
        gaps.append(("tl", "ARTIFACT_TL_IDENTITY_MISMATCH"))
    return gaps


def require_export_ready(module: Module, record: ChainRecord | None = None, *, raising: bool = True) -> dict:
    """Judge the ACTUAL installed program, never self-reported metadata."""
    record = record if record is not None else getattr(module, "metadata", {}).get(CHAIN_KEY)
    metadata = getattr(module, "metadata", {}) or {}
    gaps = []
    if record is None:
        gaps.append(("chain", "CHAIN_NOT_RUN"))
    else:
        entry = _entry_prim(module)
        if entry is None:
            gaps.append(("entry", "ACTUAL_ENTRY_MISSING"))
        else:
            actual_entry_identity = _function_identity(entry)
            if not record.entry_identity or record.entry_identity != actual_entry_identity:
                gaps.append(("entry", "ENTRY_IDENTITY_MISMATCH"))
        if record.ir_identity != ir_identity(module):
            gaps.append(("chain", "STALE_IR_IDENTITY"))
        gaps.extend(_metadata_artifact_gaps(metadata, record))
        if entry is None:
            gaps.append(("tl", "TL_ENTRY_MISSING"))
        else:
            gaps.extend(_native_artifact_gaps(metadata, entry, record))
        if record.pipeline_status != "APPLIED":
            gaps.append(("chain", record.reason or "MAINLINE_NOT_APPLIED"))
        if record.applied != SINGLE_TRANSACTION:
            gaps.append(("chain", "TRANSACTION_INCOMPLETE"))
        if record.stopped_at:
            gaps.append(("chain", "CHAIN_STOPPED_AT_%s" % record.stopped_at.upper()))
    # Layered checks, replacing the earlier unconditional
    # ("tl", "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"). That one byte made
    # every refusal indistinguishable from every other, so no layer could be
    # narrowed as it landed and nobody could see what was still missing; a
    # permanent refusal is a design defect, not fail-closed behaviour. The layers
    # below each pass or fail on their own, by READING REAL OBJECTS AND CALLING
    # REAL JUDGMENT FUNCTIONS (see passes/gate_layers.py):
    #   (i)   the installed candidate has a verifier-registered receipt
    #   (ii)+(iii) the TL artifact belongs to THIS entry and the verdict
    #         read_actual_tl recomputes from its own source/parse is VERIFIED
    #   (iv)  the ABI's counts and trailing output parameter names are the entry's
    #   (v)   the completion's obligations have matching sync statements in the
    #         candidate body
    # plus the binary layer, whose three states stay distinct: not compiled,
    # toolchain blocked, or compiled. Nothing here accepts a status byte, a dict
    # standing in for a verdict, or a self-reported identity; the verdict is
    # recomputed rather than read from metadata, so a cached answer cannot
    # authorize a program that has since changed its bytes.
    artifact = metadata.get(TL_ARTIFACT_KEY)
    artifact = artifact if isinstance(artifact, NativeArtifact) else None
    completion = metadata.get(COMPLETION_KEY)
    gaps.extend(gate_layers.plan_gaps(metadata, record, PLAN_KEY))
    gaps.extend(gate_layers.completion_gaps(metadata, COMPLETION_KEY, PLAN_KEY))
    gaps.extend(gate_layers.install_gaps(entry))
    gaps.extend(gate_layers.tl_conformance_gaps(artifact, entry))
    gaps.extend(gate_layers.binary_gaps(artifact))
    gaps.extend(gate_layers.abi_gaps(artifact, entry))
    gaps.extend(gate_layers.event_gaps(entry, completion))
    verdict = {"ready": not gaps, "gaps": [list(gap) for gap in gaps],
               "pipeline": getattr(record, "pipeline_status", "") if record else ""}
    if gaps and raising:
        first = gaps[0]
        raise ExportRefusal(
            "EXPORT_NOT_RUNNABLE",
            "a runnable artifact requires the actual installed entry and real parsed TL/ABI "
            "conformance; %d gap(s): %s"
            % (len(gaps), ", ".join("%s:%s" % (kind, code) for kind, code in gaps)),
            missing=tuple(code for _kind, code in gaps), gate=first[0])
    return verdict


@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class RunAscendNativeMainline(ModulePass):
    """Record the one transaction; never publish metadata-only APPLIED."""

    name = "ascend.run_native_mainline"

    def run(self, module: Module) -> Module:
        record = ChainRecord(ir_identity=ir_identity(module))
        try:
            plan = compile_mainline(module)
        except MainlineNotImplemented as error:
            record = replace(record, stopped_at="freeze", reason=str(error))
        else:
            entry = _entry_prim(module)
            record = replace(
                record,
                entry_identity=_function_identity(entry) if entry is not None else "",
                pipeline_status=str(getattr(plan, "pipeline_status", "REFUSED")),
                applied=tuple(getattr(plan, "applied", ())),
                plan_identity=str(getattr(plan, "plan_identity", "")),
                completion_identity=str(getattr(plan, "completion_identity", "")),
                storage_identity=str(getattr(plan, "storage_identity", "")),
                tl_identity=str(getattr(plan, "tl_identity", "")),
                stopped_at=str(getattr(plan, "stopped_at", "")),
                reason=str(getattr(plan, "reason", "")),
            )
        module.metadata[CHAIN_KEY] = record
        module.metadata[EXPORT_GATE_KEY] = require_export_ready(module, record, raising=False)
        return module
