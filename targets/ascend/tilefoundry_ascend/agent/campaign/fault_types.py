"""Typed fault planes, L3 resolution states, and structured fault payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum


class FaultPlane(str, Enum):
    """Canonical fault-plane taxonomy separating candidate errors from infra faults."""

    NONE = "NONE"
    NOOP_IDENTICAL = "NOOP_IDENTICAL"
    SYNTAX_AST = "SYNTAX_AST"
    STATIC_CONTRACT = "STATIC_CONTRACT"
    COMPILER_LOWERING = "COMPILER_LOWERING"
    CANDIDATE_RUNTIME = "CANDIDATE_RUNTIME"
    CANDIDATE_NUMERIC = "CANDIDATE_NUMERIC"
    INFRA_OCCUPANCY = "INFRA_OCCUPANCY"
    INFRA_DEVICE_FAULT = "INFRA_DEVICE_FAULT"
    LLM_TRANSPORT = "LLM_TRANSPORT"
    UNMODELED_FAULT = "UNMODELED_FAULT"

    def penalizes_bandit(self) -> bool:
        """Return True when the fault should count against bandit/rule exploration."""
        return self not in {
            FaultPlane.NONE,
            FaultPlane.INFRA_OCCUPANCY,
            FaultPlane.INFRA_DEVICE_FAULT,
            FaultPlane.LLM_TRANSPORT,
        }


class L3Resolution(str, Enum):
    """Terminal L3 acceptance resolution for a candidate round."""

    NOT_RUN = "NOT_RUN"
    PROMOTED = "PROMOTED"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    INFRA_DEFERRED = "INFRA_DEFERRED"


_STAGE_PLANE_ALIASES: dict[str, FaultPlane] = {
    "STAGE_A_STATIC": FaultPlane.STATIC_CONTRACT,
    "STAGE_B_COMPILE": FaultPlane.COMPILER_LOWERING,
    "INFRA_TOOLCHAIN": FaultPlane.INFRA_DEVICE_FAULT,
    "COST_PRESCREEN": FaultPlane.STATIC_CONTRACT,
}


def coerce_fault_plane(raw_plane: object) -> FaultPlane:
    """Resolve a FaultPlane from enum values or stage producer labels."""
    if isinstance(raw_plane, FaultPlane):
        return raw_plane
    key = str(raw_plane or "").strip().upper()
    if not key:
        return FaultPlane.NONE
    if key in _STAGE_PLANE_ALIASES:
        return _STAGE_PLANE_ALIASES[key]
    try:
        return FaultPlane(key)
    except ValueError:
        return FaultPlane.UNMODELED_FAULT


@dataclass(frozen=True)
class TypedFault:
    """Structured fault record emitted directly by compiler, runtime, and gate stages."""

    plane: FaultPlane
    code: str
    failed_tier: str
    detail: str
    failed_cases: tuple[int, ...] = ()
    retriable: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize fault record into a JSON-compatible dictionary."""
        return {
            "plane": self.plane.value,
            "code": self.code,
            "failed_tier": self.failed_tier,
            "detail": self.detail,
            "failed_cases": list(self.failed_cases),
            "retriable": bool(self.retriable),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TypedFault:
        """Reconstruct TypedFault from a serialized dictionary."""
        raw_plane = payload.get("plane", FaultPlane.UNMODELED_FAULT.value)
        plane = coerce_fault_plane(raw_plane) if raw_plane != "" else FaultPlane.UNMODELED_FAULT
        raw_cases = payload.get("failed_cases", ())
        cases: list[int] = []
        if isinstance(raw_cases, Sequence) and not isinstance(raw_cases, (str, bytes)):
            for item in raw_cases:
                if isinstance(item, bool):
                    continue
                if isinstance(item, (int, float)):
                    cases.append(int(item))
                elif isinstance(item, str):
                    try:
                        cases.append(int(item.strip()))
                    except ValueError:
                        continue
        return cls(
            plane=plane,
            code=str(payload.get("code", "")),
            failed_tier=str(payload.get("failed_tier", "")),
            detail=str(payload.get("detail", "")),
            failed_cases=tuple(cases),
            retriable=bool(payload.get("retriable", False)),
        )


class TypedFaultError(RuntimeError):
    """Exception carrying a structured TypedFault across stage boundaries."""

    def __init__(self, fault: TypedFault, message: str | None = None) -> None:
        self.fault: TypedFault = fault
        super().__init__(message if message is not None else (fault.detail or fault.code))


__all__ = [
    "FaultPlane",
    "L3Resolution",
    "TypedFault",
    "TypedFaultError",
    "coerce_fault_plane",
]
