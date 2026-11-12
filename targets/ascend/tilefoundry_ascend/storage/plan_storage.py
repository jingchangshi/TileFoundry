"""Orchestration: resolved plan + completion plan -> verified physical storage placement.

The order follows spec 3.7: versions are already resolved (P3-02), lifetimes come from the
completion program (P4-01), reuse is attempted only where those lifetimes allow it, and the
result is placed and re-checked. Each step can refuse on its own, and the refusal names the
input that was missing rather than returning a placement nobody can justify.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .model import STORAGE_SCHEMA, StorageFacts, StorageRefusal, facts_for
from .placement import STATUS_PLACED, PlacementReport, StoragePlacement
from .plan_placement import plan_storage
from .verify import STATUS_VERIFIED, StorageVerdict, verify_storage

#: Where a verified storage placement travels on Module.metadata, and the key the chain and
#: the export gate read. A declared place, not a side channel.
STORAGE_PLACEMENT_KEY = "ascend.storage_placement"


@dataclass(frozen=True)
class StoragePlan:
    """The placement plus everything needed to re-judge it later."""

    schema: str
    placement: StoragePlacement
    verdict: StorageVerdict
    facts: StorageFacts
    completion_digest: str = ""
    diagnostics: tuple = ()

    @property
    def status(self) -> str:
        return self.verdict.status

    @property
    def verified(self) -> bool:
        return self.verdict.verified

    @property
    def code(self) -> str:
        return self.verdict.code

    @property
    def placements(self) -> dict:
        return self.placement.placements

    @property
    def peak_bytes(self) -> int:
        return self.verdict.peak_bytes

    def as_dict(self):
        return {"schema": self.schema, "status": self.status, "code": self.code,
                "facts": self.facts.as_dict(),
                "completion_digest": self.completion_digest,
                "peak_bytes": self.peak_bytes,
                "spaces": {name: usage.as_dict()
                           for name, usage in (self.verdict.spaces or {}).items()},
                "placement": self.placement.as_dict(),
                "diagnostics": list(self.diagnostics)}


def _completion_digest(completion) -> str:
    if completion is None:
        return ""
    report = getattr(completion, "report", None)
    for holder in (report, completion):
        value = getattr(holder, "result_digest", "") if holder is not None else ""
        if value:
            return str(value)
    return ""


def build_storage_plan(plan, completion, *, facts=None, capacities=None, scratch_roots=(),
                       reserved_roots=(), profile_id=None) -> StoragePlan:
    """Derive, place and verify physical storage. A refusal is RETURNED, not raised."""
    facts = facts or facts_for(profile_id)
    digest = _completion_digest(completion)
    trace = getattr(completion, "trace", None) if completion is not None else None
    if trace is None:
        placement = StoragePlacement(
            schema=STORAGE_SCHEMA, entries=(), facts=facts, input_digest="",
            completion_digest=digest,
            report=PlacementReport("REFUSED", "STORAGE_COMPLETION_ABSENT",
                                   "physical reuse depends on asynchronous completion, so a"
                                   " placement without the completion program would be"
                                   " reuse decided by instruction order -- the defect this"
                                   " stage exists to remove"))
        verdict = StorageVerdict("FAILED", "STORAGE_COMPLETION_ABSENT",
                                 placement.report.detail)
        return StoragePlan(schema=STORAGE_SCHEMA, placement=placement, verdict=verdict,
                           facts=facts, completion_digest=digest)

    placement = plan_storage(plan, trace, facts=facts, capacities=capacities,
                             scratch_roots=scratch_roots, reserved_roots=reserved_roots,
                             profile_id=profile_id, completion_digest=digest)
    verdict = verify_storage(plan, placement, completion=completion, facts=facts)
    if placement.report.status == STATUS_PLACED and verdict.status != STATUS_VERIFIED:
        # The placement computed cleanly but did not survive verification: report the
        # verifier's code, because that is the reason it is unusable.
        pass
    return StoragePlan(schema=STORAGE_SCHEMA, placement=placement, verdict=verdict,
                       facts=facts, completion_digest=digest,
                       diagnostics=tuple(getattr(placement, "diagnostics", ()) or ()))


def attach_storage_plan(plan, storage_plan) -> "object":
    """Attach a VERIFIED placement to the resolved plan as real buffer offsets.

    A refused placement is not attached: the plan is returned unchanged, so the chain stops
    at layout rather than carrying addresses nobody verified.
    """
    if storage_plan is None or not storage_plan.verified:
        return plan
    from ..plan.resolved_plan import resolve_buffer_layout

    facts = {}
    for entry in storage_plan.placement.entries:
        facts[entry.root] = {"versions": entry.slot_count, "alignment": 1,
                             "version_stride_bytes": entry.version_stride_bytes,
                             "arena_id": "%s.%s" % (entry.space, entry.owner or "device"),
                             "base_offset_bytes": entry.base_offset_bytes}
    return resolve_buffer_layout(plan, facts)


def require_storage_verified(storage_plan):
    """The raising form the export path uses."""
    if storage_plan is None or not storage_plan.verified:
        code = getattr(storage_plan, "code", "") or "STORAGE_PLACEMENT_UNVERIFIED"
        detail = ""
        if storage_plan is not None:
            detail = getattr(getattr(storage_plan, "verdict", None), "detail", "")
        from ..plan.resolved_plan import PlanError

        raise PlanError(code, detail or "physical storage placement is not verified",
                        "storage", "VERIFIED placement", getattr(storage_plan, "status", None))
    return storage_plan


__all__ = ["STORAGE_PLACEMENT_KEY", "StoragePlan", "build_storage_plan",
           "attach_storage_plan", "require_storage_verified"]
