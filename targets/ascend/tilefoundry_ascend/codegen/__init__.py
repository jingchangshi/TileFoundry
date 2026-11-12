"""Native TL emission surface, with the fixed two-key PassConfig contract.

The old single-key profile and static micro-case bypass were archived.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..plan.mainline import MainlineNotImplemented

__all__ = ["NativeArtifact", "PASS_CONFIGS", "emit_supported_tl"]

PASS_CONFIGS = (("tl.enable_auto_schedule", False),
                ("tl.disable_shared_memory_reuse", True))


@dataclass(frozen=True)
class NativeArtifact:
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


def emit_supported_tl(module, plan, completion, storage, target):
    raise MainlineNotImplemented("TL_EMIT_NOT_IMPLEMENTED")
