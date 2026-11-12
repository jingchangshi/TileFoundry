"""X0-X2 codegen boundary/runtime contract (CPU only).

Design: architect_x_codegen_runtime_module_design_v1.md (frozen). X does NOT create
a new compiler and does NOT touch the device. It defines:
  * LOGICAL_TL_RECOMPILE vs RESOLVED_PLAN_PRESERVING;
  * a Plan-preserve request is refused if TL re-schedules/overwrites the Plan;
  * four identities source map / ABI / CCE / binary are separate, with UNOBSERVED
    allowed only for non-execution CPU generation;
  * compile/bind/invoke are explicitly CODEGEN_CONTRACT_UNIMPLEMENTED until X3/X4.

Only R/C VERIFIED inputs may reach X. emit_for_codegen is the sole plan boundary.
X3 consumes the H4 identity chain for TL_ASCEND_C_SOURCE_PATH: source+CCE+load must be
OBSERVED with hashes and binary must be explicitly UNOBSERVED_BY_BUILD; a fabricated
binary OBSERVED or a generic UNOBSERVED is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


class CodegenError(RuntimeError):
    def __init__(self, code, message, source_id="", expected=None, actual=None, phase="codegen"):
        self.code = code
        self.phase = phase
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (source=%r expected=%r actual=%r)"
                         % (code, message, source_id, expected, actual))

    def as_dict(self):
        return {"code": self.code, "phase": self.phase, "source_id": self.source_id,
                "expected": self.expected, "actual": self.actual}


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps({"value": value}, sort_keys=True, default=str).encode()).hexdigest()


LOGICAL_TL_RECOMPILE = "LOGICAL_TL_RECOMPILE"
RESOLVED_PLAN_PRESERVING = "RESOLVED_PLAN_PRESERVING"
MODES = (LOGICAL_TL_RECOMPILE, RESOLVED_PLAN_PRESERVING)
TL_ASCEND_C_SOURCE_PATH = "TL_ASCEND_C_SOURCE_PATH"
BINARY_UNOBSERVED_BY_BUILD = "UNOBSERVED_BY_BUILD"
IDENTITY_KEYS = ("source", "cce", "binary", "load")


@dataclass(frozen=True)
class SourceIdentity:
    source_kind: str
    tir_digest: str
    source_map: dict
    adapter_version: str
    source_hash: Optional[str] = None
    handwritten_twin: bool = False


# THE FLOAT DTYPES A KERNEL ARGUMENT MAY DECLARE, keyed by every spelling a caller uses.
#
# This list is deliberately a REGISTRY rather than an inline condition. G42 requires the ABI to
# support FP16/BF16 without an RMSNorm-shaped assumption, and the previous version hardcoded an
# f32/bf16 pair because the only caller was an RMSNorm kernel. A registry keeps the supported
# set explicit and makes adding a dtype a one-line change instead of a hunt through conditions.
FLOAT_ARG_BYTES = {
    "f16": 2, "float16": 2, "half": 2, "fp16": 2,
    "bf16": 2, "bfloat16": 2,
    "f32": 4, "float32": 4, "float": 4, "fp32": 4,
    "f64": 8, "float64": 8, "double": 8, "fp64": 8,
}

# Scalar arguments carry a value, not a buffer, so they have no byte size of their own.
#
# FLOAT SCALARS ARE INCLUDED ON PURPOSE. An epsilon, a scale, a normalisation reciprocal -- the
# most common runtime scalar in this target is a float, and leaving them out would have made the
# "general ABI" reject exactly the RMSNorm-shaped kernel it was meant to stop special-casing.
SCALAR_ARG_DTYPES = {
    "i8", "int8", "u8", "uint8", "i16", "int16", "u16", "uint16",
    "i32", "int32", "u32", "uint32", "i64", "int64", "u64", "uint64",
    "bool", "index",
} | set(FLOAT_ARG_BYTES)

SCALAR_ARG_KINDS = ("scalar", "runtime", "value")


def arg_is_scalar(arg):
    """Is this ABI argument a scalar? SCALARS ARE NOT BUFFERS AND MUST NOT BE SIZED AS ONE."""
    kind = str(_get(arg, "kind", "")).lower()
    if kind in SCALAR_ARG_KINDS:
        return True
    if kind in ("in", "out", "inout"):
        return False
    # An argument that declares no byte size but does declare a scalar dtype is a scalar. The
    # old validator called that an error, which forced every caller to invent a buffer size for
    # a value that has none -- the RMSNorm-specific assumption this removes.
    return str(_get(arg, "dtype", "")).lower() in SCALAR_ARG_DTYPES


@dataclass(frozen=True)
class KernelABI:
    """The kernel's calling convention, GENERAL rather than shaped around one kernel.

    What was RMSNorm-specific here, and why each one blocked a legitimate kernel:

      * a SINGLE 'output_index' -- a kernel with two outputs (scores and log-sum-exp, a
        normalisation's mean and variance) could not be described at all. Outputs are now the
        set of args whose kind is out/inout, and 'output_index' survives only as a convenience
        for the overwhelmingly common single-output case.
      * 'eps_f32_bits' as a required positional field -- an epsilon is RMSNorm's parameter, not
        the ABI's. Runtime scalars are ordinary args of kind 'scalar', passed positionally.
      * an f32/bf16-only dtype whitelist -- FP16 is a first-class Ascend dtype. Dtypes now come
        from a registry, and the check is "is it a dtype we can size", not "is it this kernel's".
      * 'nbytes > 0 required for every arg' -- a runtime scalar has no byte size. Requiring one
        meant a scalar could not be expressed without pretending it was a buffer.
    """

    args: tuple
    output_index: int = -1
    eps_f32_bits: Optional[int] = None
    grid: dict = field(default_factory=dict)

    def buffer_args(self):
        return tuple(arg for arg in self.args if not arg_is_scalar(arg))

    def scalar_args(self):
        return tuple(arg for arg in self.args if arg_is_scalar(arg))

    def outputs(self):
        """Every out/inout argument, in declaration order. Not just one."""
        return tuple(arg for arg in self.args
                     if str(_get(arg, "kind", "")).lower() in ("out", "inout"))

    def validate(self):
        if not isinstance(self.args, (tuple, list)) or not self.args:
            raise CodegenError("X_ABI_MISMATCH", "ABI must contain ordered args",
                               "args", "nonempty tuple", self.args)
        names = [str(_get(arg, "name", "")) for arg in self.args]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise CodegenError("X_ABI_MISMATCH", "ABI arg names must be unique/nonempty",
                               "args", names, names)

        outputs = self.outputs()
        if not outputs:
            raise CodegenError("X_ABI_MISMATCH",
                               "ABI declares no out/inout arg; a kernel must produce something",
                               "args", "at least one out/inout", names)
        if self.output_index >= 0:
            # The index is OPTIONAL (a multi-output kernel has no single primary), but when it is
            # given it must still name a real output rather than an input.
            if not (0 <= int(self.output_index) < len(self.args)):
                raise CodegenError("X_ABI_MISMATCH", "output_index out of range",
                                   "output_index", (0, len(self.args) - 1), self.output_index)
            chosen = self.args[int(self.output_index)]
            if str(_get(chosen, "kind")).lower() not in ("out", "inout"):
                raise CodegenError("X_ABI_MISMATCH", "output arg must be out/inout",
                                   "output_index", "out/inout", _get(chosen, "kind"))

        for arg in self.args:
            name = str(_get(arg, "name"))
            dtype = str(_get(arg, "dtype", "")).lower()
            if not dtype:
                raise CodegenError("X_ABI_MISMATCH", "arg dtype must be explicit",
                                   name, "an explicit dtype", dtype)
            if arg_is_scalar(arg):
                if dtype not in SCALAR_ARG_DTYPES:
                    raise CodegenError(
                        "X_ABI_MISMATCH",
                        "a scalar arg declares a non-scalar dtype %r; a scalar carries a value,"
                        " not a buffer element" % dtype, name, sorted(SCALAR_ARG_DTYPES), dtype)
                # A scalar's byte size is DERIVED from its dtype; it is not supplied, and asking
                # for one was the RMSNorm-shaped assumption this ABI removes.
                continue
            if dtype not in FLOAT_ARG_BYTES:
                raise CodegenError(
                    "X_ABI_MISMATCH", "arg dtype must be a float dtype this target can size",
                    name, sorted(FLOAT_ARG_BYTES), dtype)
            nbytes = _get(arg, "nbytes")
            if nbytes is None or int(nbytes) <= 0:
                raise CodegenError("X_ABI_MISMATCH", "buffer arg nbytes must be explicit",
                                   name, ">0", nbytes)
            expected = FLOAT_ARG_BYTES[dtype]
            count = _get(arg, "count")
            if count is not None and int(nbytes) != int(count) * expected:
                raise CodegenError(
                    "X_ABI_MISMATCH",
                    "arg declares %s bytes for %s element(s) of %s, which is %s bytes"
                    % (nbytes, count, dtype, int(count) * expected),
                    name, int(count) * expected, nbytes)
        return True


@dataclass(frozen=True)
class AdapterRequest:
    schema: str
    run_id: str
    case_id: str
    workload_id: str
    mode: str
    tf_source_ref: dict
    tf_hir_ref: dict
    tir_ref: dict
    problem_ref: dict
    plan_ref: dict
    c_ref: dict
    env_ref: dict
    input_ref: dict
    source_identity: SourceIdentity
    abi: KernelABI
    plan_preserved: bool


@dataclass(frozen=True)
class TLModule:
    source: Optional[str]
    module: Optional[str]
    source_map: dict
    plan_preserved: bool
    plan_claims: dict
    source_status: str = "SOURCE_MAPPED"


@dataclass(frozen=True)
class CompiledArtifactBundle:
    cce: dict
    binary: dict
    load: dict
    source_map: dict
    status: str = "COMPILED_UNOBSERVED"


def validate_adapter_request(tir, plan, C_report, env_ref, mode, *, source_map=None,
                             abi=None, run_ref=None) -> AdapterRequest:
    if mode not in MODES:
        raise CodegenError("X_MODE_INVALID", "unknown codegen mode", "mode", MODES, mode)
    run_ref = run_ref or {}
    if mode == RESOLVED_PLAN_PRESERVING:
        state = _get(plan, "state", {}) if plan is not None else {}
        if str(_get(state, "application_status")) != "APPLIED":
            raise CodegenError("X_PLAN_NOT_READY", "R plan is not APPLIED",
                               "application_status", "APPLIED", _get(state, "application_status"))
        if str(_get(C_report, "status", "")).upper() != "VERIFIED":
            raise CodegenError("COMPLETION_NOT_VERIFIED", "C completion is not VERIFIED",
                               "C", "VERIFIED", _get(C_report, "status"))
        required = int(_get(C_report, "required", 0) or 0)
        covered = int(_get(C_report, "covered", 0) or 0)
        if required <= 0 or covered != required:
            raise CodegenError("COMPLETION_NOT_VERIFIED", "C coverage is incomplete",
                               "C", "required==covered>0", [required, covered])
    else:
        if run_ref.get("claim_plan") or run_ref.get("plan_ref"):
            raise CodegenError("X_LOGICAL_PLAN_CLAIM_FORBIDDEN",
                               "LOGICAL_TL_RECOMPILE cannot claim a resolved Plan",
                               "mode", "logical TL recompile evidence only", run_ref)
    if not tir or not env_ref:
        raise CodegenError("X_SOURCE_IDENTITY_MISMATCH", "tir/env identity missing",
                           "request", "nonempty tir/env", [tir, env_ref])
    source_map = source_map or {}
    if not source_map:
        raise CodegenError("X_SOURCE_COVERAGE_MISMATCH", "no TL source map supplied",
                           "source_map", "nonempty map", None)
    if abi is None:
        raise CodegenError("X_ABI_MISMATCH", "no KernelABI supplied", "abi", "KernelABI", None)
    if isinstance(abi, dict):
        abi = KernelABI(tuple(abi.get("args", ())), int(abi.get("output_index", -1)),
                        abi.get("eps_f32_bits"), abi.get("grid", {}))
    abi.validate()
    identity = SourceIdentity(str(run_ref.get("source_kind", "TF_TYPED_TIR")),
                              str(run_ref.get("tir_digest", _digest(tir))),
                              dict(source_map), str(run_ref.get("adapter_version", "x0-x2")),
                              run_ref.get("source_hash"))
    return AdapterRequest(
        schema="x-runtime/1", run_id=str(run_ref.get("run_id", "x-run")),
        case_id=str(run_ref.get("case_id", "case")), workload_id=str(run_ref.get("workload_id", "vadd")),
        mode=mode, tf_source_ref=dict(run_ref.get("tf_source_ref", {})),
        tf_hir_ref=dict(run_ref.get("tf_hir_ref", {})), tir_ref=dict(run_ref.get("tir_ref", {})),
        problem_ref=dict(run_ref.get("problem_ref", {})), plan_ref=dict(run_ref.get("plan_ref", {})),
        c_ref=dict(run_ref.get("c_ref", {})), env_ref=dict(env_ref),
        input_ref=dict(run_ref.get("input_ref", {})), source_identity=identity, abi=abi,
        plan_preserved=bool(run_ref.get("plan_preserved", mode == RESOLVED_PLAN_PRESERVING)))


def lower_to_tilelang(request: AdapterRequest, actual_tir, source_maps) -> TLModule:
    if not actual_tir or not source_maps:
        raise CodegenError("X_SOURCE_COVERAGE_MISMATCH", "actual TIR/source map missing",
                           "lower", "nonempty TIR+map", [actual_tir, source_maps])
    statements = source_maps.get("statements", {}) or {}
    if not statements:
        raise CodegenError("X_SOURCE_COVERAGE_MISMATCH", "no effect statement mapped",
                           "source_map.statements", "nonempty", source_maps)
    plan_preserved = bool(source_maps.get("plan_preserved", False))
    if request.mode == RESOLVED_PLAN_PRESERVING and not plan_preserved:
        raise CodegenError("X_PLAN_PRESERVATION_MISMATCH",
                           "TL would reschedule/overwrite the Plan but preserve was requested",
                           "mode", "plan_preserved=true", plan_preserved)
    return TLModule(source=None, module=None, source_map=dict(source_maps),
                    plan_preserved=plan_preserved, plan_claims=dict(source_maps.get("plan_claims", {})),
                    source_status="SOURCE_MAPPED")


def verify_adapter_correspondence(request: AdapterRequest, tl_module: TLModule,
                                  source_map, abi, *, claim_pass=False,
                                  identity_chain=None, claim_path=TL_ASCEND_C_SOURCE_PATH) -> dict:
    source_map = source_map or {}
    statements = source_map.get("statements", {}) or {}
    if not statements:
        raise CodegenError("X_SOURCE_COVERAGE_MISMATCH", "source map has no statements",
                           "source_map", "nonempty", source_map)
    if request.mode == RESOLVED_PLAN_PRESERVING and not tl_module.plan_preserved:
        raise CodegenError("X_PLAN_PRESERVATION_MISMATCH",
                           "TL module does not preserve the resolved Plan",
                           "plan_preserved", True, tl_module.plan_preserved)
    if not isinstance(abi, KernelABI):
        raise CodegenError("X_ABI_MISMATCH", "runtime ABI must be typed at the bind boundary",
                           "abi", "KernelABI", type(abi).__name__)
    chain = None
    if identity_chain is not None:
        chain = validate_h4_identity_chain(identity_chain, claim_path=claim_path,
                                           claim_pass=claim_pass)
        cce = {"status": chain["identities"]["cce"]["status"]}
        binary = {"status": chain["identities"]["binary"]["status"]}
        load = {"status": chain["identities"]["load"]["status"]}
    else:
        cce = source_map.get("cce") or {}
        binary = source_map.get("binary") or {}
        load = source_map.get("load") or {}
    missing = [name for name, value in (("cce", cce), ("binary", binary), ("load", load))
               if not value or str(_get(value, "status", "")).upper() == "UNOBSERVED"]
    if claim_pass and missing:
        raise CodegenError("X_ARTIFACT_IDENTITY_MISSING",
                           "PASS claimed without observed CCE/binary/load identity",
                           "artifacts", "OBSERVED cce/binary/load", missing)
    return {"status": "SOURCE_MAPPED", "source_map": True, "abi": True,
            "plan_preserved": bool(tl_module.plan_preserved),
            "identities": {"source": "MAPPED", "cce": str(_get(cce, "status", "UNOBSERVED")),
                           "binary": str(_get(binary, "status", "UNOBSERVED")),
                           "load": str(_get(load, "status", "UNOBSERVED"))},
            "missing_execution_identities": missing, "claim_pass": bool(claim_pass),
            "identity_chain": chain}


TL_ASCEND_C_SOURCE_PATH = "TL_ASCEND_C_SOURCE_PATH"
BINARY_UNOBSERVED_BY_BUILD = "UNOBSERVED_BY_BUILD"
IDENTITY_KEYS = ("source", "cce", "binary", "load")


def _identity_status(value):
    if not isinstance(value, dict):
        return "MISSING", None, ""
    status = str(value.get("status", "MISSING")).upper()
    sha = value.get("sha256")
    reason = str(value.get("reason") or "")
    if status == "OBSERVED" and not sha:
        return "OBSERVED_WITHOUT_HASH", sha, reason
    return status, sha, reason


def normalize_h4_identity_chain(identities):
    identities = dict(identities or {})
    normalized = {}
    for key in IDENTITY_KEYS:
        source = identities.get(key)
        status, sha, reason = _identity_status(source)
        if key == "binary" and status in ("UNOBSERVED", "UNOBSERVED_BY_BUILD"):
            lowered = reason.lower()
            if status == "UNOBSERVED_BY_BUILD" or ("artifact" in lowered and
                                                    ("none" in lowered or "not expose" in lowered)):
                status = BINARY_UNOBSERVED_BY_BUILD
            else:
                status = "UNOBSERVED_UNNAMED"
        normalized[key] = {"status": status, "sha256": sha, "reason": reason}
    return normalized


def validate_h4_identity_chain(identities, *, claim_path=TL_ASCEND_C_SOURCE_PATH,
                               claim_pass=False):
    if claim_path != TL_ASCEND_C_SOURCE_PATH:
        raise CodegenError("X_CLAIM_PATH_UNSUPPORTED", "unknown identity claim path",
                           "claim_path", TL_ASCEND_C_SOURCE_PATH, claim_path)
    chain = normalize_h4_identity_chain(identities)
    missing = [key for key in ("source", "cce", "load")
               if chain[key]["status"] != "OBSERVED" or not chain[key]["sha256"]]
    binary = chain["binary"]
    if claim_pass:
        if missing:
            raise CodegenError("X_ARTIFACT_IDENTITY_MISSING",
                               "claim requires observed source/CCE/load identities",
                               "identity_chain", "OBSERVED source/cce/load", missing)
        if binary["status"] == "OBSERVED":
            raise CodegenError("X_BINARY_IDENTITY_UNEXPECTED",
                               "binary OBSERVED is not valid for this TL build; do not fabricate binary hash",
                               "binary", BINARY_UNOBSERVED_BY_BUILD, "OBSERVED")
        if binary["status"] != BINARY_UNOBSERVED_BY_BUILD:
            raise CodegenError("X_BINARY_IDENTITY_UNNAMED",
                               "binary must be explicitly UNOBSERVED_BY_BUILD with reason",
                               "binary", BINARY_UNOBSERVED_BY_BUILD, binary)
    ready = bool(not missing and binary["status"] == BINARY_UNOBSERVED_BY_BUILD)
    return {"schema": "x-identity-chain/1", "claim_path": claim_path,
            "status": "IDENTITY_CHAIN_READY" if ready else "IDENTITY_CHAIN_PARTIAL",
            "identities": chain, "missing_required": missing,
            "binary_status": binary["status"], "claim_pass": bool(claim_pass)}


def identities_from_h4_result(result):
    return dict((result or {}).get("identities") or (result or {}).get("artifacts") or {})

def compile_tilelang(module, abi, env_context, compile_policy):
    from .native_aiv import NativeAIVArtifact
    if isinstance(module, NativeAIVArtifact):
        import importlib.util
        from pathlib import Path
        import tilelang
        if hashlib.sha256(module.source.encode()).hexdigest() != module.source_sha256:
            raise CodegenError("X_SOURCE_IDENTITY_MISMATCH", "native source digest changed")
        if abi != module.abi or compile_policy != {"tl.enable_auto_schedule": False}:
            raise CodegenError("X_PLAN_PRESERVATION_MISMATCH",
                               "native ABI and explicit compile policy must match the artifact")
        path = Path(env_context["source_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(module.source)
        spec = importlib.util.spec_from_file_location("native_aiv_" + module.source_sha256, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return tilelang.compile(loaded.main, target="ascend", out_idx=None,
                                pass_configs=compile_policy)
    raise CodegenError("CODEGEN_CONTRACT_UNIMPLEMENTED",
                       "real TL compile is X3; no CCE/binary emitted in X0-X2",
                       "compile_tilelang", "implemented X3 compile", None)


def bind_kernel(bundle, abi, actual_inputs, selected_device):
    raise CodegenError("CODEGEN_CONTRACT_UNIMPLEMENTED",
                       "bind is X4; no device binding in X0-X2",
                       "bind_kernel", "implemented X4 bind", None)


def invoke_kernel(bound_kernel, run_context):
    raise CodegenError("CODEGEN_CONTRACT_UNIMPLEMENTED",
                       "invoke is X4; no device launch in X0-X2",
                       "invoke_kernel", "implemented X4 invoke", None)


def synchronize_and_readback(handles, run_context):
    raise CodegenError("CODEGEN_CONTRACT_UNIMPLEMENTED",
                       "sync/readback is X4; no device in X0-X2",
                       "synchronize_and_readback", "implemented X4 readback", None)


def emit_for_codegen(plan, codegen_binding=None):
    from ..plan.resolved_plan import emit_for_codegen as _plan_emit
    return _plan_emit(plan, codegen_binding)


__all__ = ["CodegenError", "SourceIdentity", "KernelABI", "AdapterRequest", "TLModule",
           "FLOAT_ARG_BYTES", "SCALAR_ARG_DTYPES", "arg_is_scalar",
           "CompiledArtifactBundle", "LOGICAL_TL_RECOMPILE", "RESOLVED_PLAN_PRESERVING",
           "validate_adapter_request", "lower_to_tilelang", "verify_adapter_correspondence",
           "compile_tilelang", "bind_kernel", "invoke_kernel", "synchronize_and_readback",
           "emit_for_codegen", "TL_ASCEND_C_SOURCE_PATH", "BINARY_UNOBSERVED_BY_BUILD",
           "IDENTITY_KEYS", "normalize_h4_identity_chain", "validate_h4_identity_chain",
           "identities_from_h4_result"]
