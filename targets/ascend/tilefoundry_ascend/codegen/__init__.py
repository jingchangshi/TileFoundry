"""Native TL emission surface, with the fixed two-key PassConfig contract.

The old single-key profile and static micro-case bypass were archived.

Frozen surface:

    emit_supported_tl(module, plan, completion, storage, target) -> NativeArtifact
    read_actual_tl(artifact, parsed_prim_func, refinement_manifest) -> ConformanceVerdict

emit_supported_tl produces the real product: TileLang source text generated from
the ACTUAL lowered TIR entry, the file the deployed builder can parse, the parsed
tvm.tirx.PrimFunc, the ABI derived from the real entry, and content identities
over those bytes. read_actual_tl reads the parsed object's own fields and
compares them item by item with what the lowered TIR states. Neither compiles a
kernel nor touches a device: parse-of-emitted-source is the observation point
this slice can prove, and post-compile refinement stays recorded as
UNOBSERVED_BY_BUILD instead of being claimed.

emit/read are an artifact layer, not an authorization. They never install,
export or bind; p3_chain keeps refusing export until the whole transaction is
verified, and the two pass-config keys below are part of the artifact identity
the compile site consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ConformanceVerdict", "NativeArtifact", "PASS_CONFIGS", "RefinementManifest",
           "compile_supported_artifact", "emit_supported_tl", "read_actual_tl"]

PASS_CONFIGS = (("tl.enable_auto_schedule", False),
                ("tl.disable_shared_memory_reuse", True))


@dataclass(frozen=True)
class NativeArtifact:
    """One emitted TL program, bound to the bytes and the parsed object it came from.

    source_identity/tl_identity are digests of the emitted source text;
    tir_identity is the identity of the lowered TIR entry the source came from.
    entry_symbol, emitted and source_path carry the structure a reader needs to
    re-parse the same file and compare it field by field. completion_identity and
    storage_identity are recorded when those stages supply them: this surface
    cannot fabricate them, and an empty one stays a named gap for the export gate.
    """

    source: str = ""
    abi: object | None = None
    parsed_prim_func: object | None = None
    source_identity: str = ""
    plan_identity: str = ""
    tir_identity: str = ""
    tl_identity: str = ""
    pass_configs: tuple = PASS_CONFIGS
    target_identity: str = ""
    toolchain_identity: str = ""
    entry_symbol: str = ""
    emitted: object | None = None
    source_path: str = ""
    completion_identity: str = ""
    storage_identity: str = ""
    #: The real Ascend C text the deployed source-level codegen produced from
    #: this kernel (lower_with_context(enable_device_compile=False) +
    #: inspect_source()). Second-layer evidence: ascend_identity digests THIS
    #: text and is never equal to source_identity/tl_identity, which digest the
    #: TileLang source text. Neither claims a compiled binary or a device run.
    ascend_source: str = ""
    ascend_identity: str = ""
    #: The binary layer, present only when the compile site ran. None means no
    #: aibin was produced: it is NOT read as "compiled with no differences", and
    #: read_actual_tl reports the event/ABI lowering as unobserved because of it.
    compiled: object | None = None
    #: A BLOCKED_ENV result from the compile site: an external toolchain gap,
    #: recorded rather than hidden, and never a reason to claim completeness.
    compile_blockage: object | None = None


def emit_supported_tl(module, plan, completion, storage, target):
    from .emit import emit_supported_tl as _emit

    return _emit(module, plan, completion, storage, target)


def read_actual_tl(artifact, parsed_prim_func, refinement_manifest=None):
    from .tl_reader import read_actual_tl as _read

    return _read(artifact, parsed_prim_func, refinement_manifest)


def compile_supported_artifact(artifact, cache_directory=None):
    """Run the explicit compile site over one emitted artifact and attach it.

    The S0 criterion wants a real kernel.aibin, host executable.so, the Ascend C
    text, params.json and manifest.json. This is the only route to them and it is
    deliberate: emit_supported_tl stays source-level so a test can certify
    structure without a toolchain, and a caller that needs the binary layer asks
    for it here. A missing bisheng comes back as the artifact plus a BLOCKED_ENV
    compile_blockage - the source layer stays usable and nothing is promoted to
    complete because the rest was unavailable.
    """
    from .ascend_binary import compile_ascend_artifact
    from dataclasses import replace

    facts = compile_ascend_artifact(artifact.parsed_prim_func, artifact.pass_configs,
                                    cache_directory=cache_directory)
    if getattr(facts, "blocked", False):
        return replace(artifact, compile_blockage=facts, compiled=None)
    return replace(artifact, compiled=facts, compile_blockage=None)
