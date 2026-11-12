"""Native post-version storage plan surface.

UNKNOWN capacity is a refusal, never a footprint backfill. The old placement
constructors that treated None as fitting were archived.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..plan.mainline import MainlineNotImplemented

__all__ = ["StoragePlan", "UNKNOWN_CAPACITY", "StorageError",
           "place_resolved_versions", "require_storage_verified", "verify_storage_current"]

UNKNOWN_CAPACITY = "UNKNOWN_CAPACITY"


class StorageError(RuntimeError):
    """A named storage refusal."""


@dataclass(frozen=True)
class StoragePlan:
    entries: tuple = ()
    verified: bool = False
    plan_iteration: int = 0
    identity: str = ""


def place_resolved_versions(plan, completion, target, lowering):
    raise MainlineNotImplemented("STORAGE_PLACEMENT_NOT_IMPLEMENTED")


def verify_storage_current(plan, storage, target):
    raise MainlineNotImplemented("STORAGE_VERIFY_NOT_IMPLEMENTED")


def require_storage_verified(storage):
    raise MainlineNotImplemented("STORAGE_REQUIRED_NOT_VERIFIED")
