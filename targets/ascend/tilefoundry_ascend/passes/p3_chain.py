"""The single native Ascend mainline pass.

The export gate is fail-closed. It reads the actual installed entry and the
actual NativeArtifact produced by the TL emitter; status bytes, dicts and
self-reported identity strings are never proof. Until the real parsed PrimFunc
and ABI conformance path exists, export is explicitly NOT_IMPLEMENTED.
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
    gaps = []
    for label, key, expected in (
        ("plan", PLAN_KEY, record.plan_identity),
        ("completion", COMPLETION_KEY, record.completion_identity),
        ("storage", STORAGE_KEY, record.storage_identity),
    ):
        artifact = metadata.get(key)
        if artifact is None:
            gaps.append((label, "%s_ARTIFACT_MISSING" % label.upper()))
        elif isinstance(artifact, dict) or not _artifact_identity(artifact):
            gaps.append((label, "%s_ARTIFACT_NOT_ACTUAL" % label.upper()))
        elif not expected or expected != _artifact_identity(artifact):
            gaps.append((label, "%s_IDENTITY_MISMATCH" % label.upper()))
    return gaps


def _native_artifact_gaps(metadata, entry, record):
    """Check the ACTUAL emitted artifact; never accept a dict or a status byte."""
    gaps = []
    artifact = metadata.get(TL_ARTIFACT_KEY)
    if not isinstance(artifact, NativeArtifact):
        return [("tl", "TL_ARTIFACT_NOT_ACTUAL")]
    if not isinstance(getattr(artifact, "source", None), str) or not artifact.source.strip():
        gaps.append(("tl", "TL_SOURCE_MISSING"))
    if tuple(getattr(artifact, "pass_configs", ())) != PASS_CONFIGS:
        gaps.append(("tl", "PASS_CONFIGS_MISMATCH"))
    abi = getattr(artifact, "abi", None)
    if abi is None or isinstance(abi, dict):
        gaps.append(("tl", "ABI_NOT_ACTUAL"))
    else:
        abi_inputs = _get(abi, "input_count", None)
        abi_outputs = _get(abi, "output_count", None)
        expected_inputs = max(0, len(getattr(entry, "params", ()) or ()) - int(getattr(entry, "output_count", 0) or 0))
        if abi_inputs is None or abi_inputs != expected_inputs:
            gaps.append(("tl", "ABI_INPUT_COUNT_MISMATCH"))
        if abi_outputs is None or abi_outputs != getattr(entry, "output_count", None):
            gaps.append(("tl", "ABI_OUTPUT_COUNT_MISMATCH"))
    parsed = getattr(artifact, "parsed_prim_func", None)
    if parsed is None or not hasattr(parsed, "params") or not hasattr(parsed, "body"):
        gaps.append(("tl", "PARSED_PRIMFUNC_MISSING"))
    for field in ("source_identity", "plan_identity", "tir_identity", "tl_identity",
                  "target_identity", "toolchain_identity"):
        if not getattr(artifact, field, ""):
            gaps.append(("tl", "ARTIFACT_%s_MISSING" % field.upper()))
    if getattr(artifact, "plan_identity", "") != record.plan_identity:
        gaps.append(("tl", "ARTIFACT_PLAN_IDENTITY_MISMATCH"))
    if getattr(artifact, "tl_identity", "") != record.tl_identity:
        gaps.append(("tl", "ARTIFACT_TL_IDENTITY_MISMATCH"))
    conformance = metadata.get(TL_CONFORMANCE_KEY)
    if conformance is None or isinstance(conformance, dict):
        gaps.append(("tl", "TL_CONFORMANCE_NOT_ACTUAL"))
    else:
        if _get(conformance, "verified", False) is not True:
            gaps.append(("tl", "TL_CONFORMANCE_UNVERIFIED"))
        if _get(conformance, "parsed_prim_func", None) is not parsed:
            gaps.append(("tl", "TL_CONFORMANCE_PARSED_PRIMFUNC_MISMATCH"))
        for field in ("entry_identity", "plan_identity", "tl_identity"):
            expected = {"entry_identity": record.entry_identity,
                        "plan_identity": record.plan_identity,
                        "tl_identity": record.tl_identity}[field]
            if not expected or _get(conformance, field, "") != expected:
                gaps.append(("tl", "TL_CONFORMANCE_%s_MISMATCH" % field.upper()))
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
    # The real TL reader/parsed PrimFunc/ABI conformance path is not implemented
    # in this candidate. Be explicit rather than accepting invented evidence.
    gaps.append(("tl", "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"))
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
