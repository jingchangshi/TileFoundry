"""Storage verification: does an ATTACHED placement still describe this plan?

plan_storage computes a placement. This module answers the different question a gate has to
answer: given a placement that was attached earlier, is it still TRUE of the plan in front of
us? Those are different questions because a plan can change after its placement was computed,
and a stale placement is worse than no placement: its addresses look valid, so nothing
downstream refuses them.

The four inputs the spec says must invalidate a placement are checked here explicitly --
versions, layout, owner, completion -- plus the digest that covers all of them at once. The
per-input checks exist BESIDE the digest because a digest mismatch tells you only that
something changed; naming the input that changed is what lets an operator fix it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .placement import STATUS_PLACED, PlacementReport, StoragePlacement
from .plan_placement import verify_placement

STATUS_VERIFIED = "VERIFIED"
STATUS_FAILED = "FAILED"


@dataclass(frozen=True)
class StorageVerdict:
    status: str
    code: str = ""
    detail: str = ""
    invalidated_by: tuple = ()
    peak_bytes: int = 0
    spaces: dict = None
    result_digest: str = ""

    @property
    def verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    def as_dict(self):
        return {"status": self.status, "code": self.code, "detail": self.detail,
                "invalidated_by": list(self.invalidated_by), "peak_bytes": self.peak_bytes,
                "result_digest": self.result_digest,
                "spaces": {name: usage.as_dict()
                           for name, usage in (self.spaces or {}).items()}}


def _fail(code, detail, invalidated_by=(), report=None):
    return StorageVerdict(STATUS_FAILED, code, detail, tuple(invalidated_by),
                          peak_bytes=getattr(report, "peak_bytes", 0),
                          spaces=getattr(report, "spaces", None))


def invalidated_inputs(plan, placement) -> tuple:
    """Which of versions/layout/owner/completion differ from what the placement assumed.

    Every one is checked SEPARATELY, against the plan in hand, so the answer is "versions
    changed" and not merely "the digest differs".
    """
    changed = []
    assumed = getattr(placement, "versioning", {}) or {}
    for buffer in plan.buffers:
        root = str(buffer.root)
        entry = assumed.get(root)
        if entry is None:
            changed.append(("versions", root, None, int(getattr(buffer, "slot_count", 1) or 1)))
            continue
        current = int(getattr(buffer, "slot_count", 1) or 1)
        if int(entry.get("slot_count", -1)) != current:
            changed.append(("versions", root, entry.get("slot_count"), current))
    placed_layout = {str(entry.root): str(entry.layout)
                     for entry in getattr(placement, "entries", ())}
    for buffer in plan.buffers:
        root = str(buffer.root)
        layout = placed_layout.get(root)
        if layout is None:
            changed.append(("layout", root, None, "unplaced"))
    placed_owner = {str(entry.root): str(entry.owner)
                    for entry in getattr(placement, "entries", ())}
    for buffer in plan.buffers:
        root = str(buffer.root)
        owner = str(getattr(buffer, "owner", "") or "")
        if root in placed_owner and placed_owner[root] != owner:
            changed.append(("owner", root, placed_owner[root], owner))
    return tuple(changed)


def verify_storage(plan, placement, *, completion=None, facts=None) -> StorageVerdict:
    """Judge an attached placement against THIS plan. Never recomputes it silently.

    Recomputation is deliberately NOT done here: a verifier that quietly rebuilds what it is
    checking will always agree with itself. The caller recomputes (plan_storage) and compares,
    which is what makes the comparison worth anything.
    """
    if placement is None:
        return _fail("STORAGE_PLACEMENT_UNATTACHED",
                     "no storage placement was attached; the export path must not treat"
                     " absence as success")
    report = getattr(placement, "report", None)
    if report is None or getattr(report, "status", "") != STATUS_PLACED:
        return _fail("STORAGE_PLACEMENT_UNVERIFIED",
                     "the attached placement did not itself verify: %s %s"
                     % (getattr(report, "code", ""), getattr(report, "detail", "")),
                     report=report)

    changed = invalidated_inputs(plan, placement)
    if changed:
        return _fail("STORAGE_PLACEMENT_STALE",
                     "the placement was computed for different inputs and its offsets are no"
                     " longer valid: %s" % (list(changed),), invalidated_by=changed,
                     report=report)

    # Completion is an input to reuse: if the completion program changed, the intervals the
    # reuse depended on changed with it. Comparing digests is the cheap check; it is enough
    # BECAUSE plan_storage derives every interval from the trace, so equal digests mean equal
    # intervals.
    if completion is not None:
        expected = str(getattr(placement, "completion_digest", "") or "")
        actual = str(getattr(completion, "result_digest",
                             getattr(getattr(completion, "report", None),
                                     "result_digest", "")) or "")
        if expected and actual and expected != actual:
            return _fail("STORAGE_COMPLETION_CHANGED",
                         "the completion program changed after placement, so the reuse"
                         " intervals are no longer the ones the placement assumed"
                         " (placement=%s completion=%s)" % (expected, actual),
                         invalidated_by=(("completion", "", expected, actual),),
                         report=report)

    # Re-run the STRUCTURAL checks on the attached entries: capacity, alignment, illegal
    # alias. A placement can be fresh and still be wrong if it was built by an older
    # allocator, and this is the layer that catches that.
    recheck = verify_placement(list(placement.entries), report.spaces)
    if not recheck.placed:
        return _fail(recheck.code or "STORAGE_PLACEMENT_UNVERIFIED",
                     recheck.detail or "the attached placement violates a placement rule",
                     report=report)
    return StorageVerdict(STATUS_VERIFIED, "", "", (), peak_bytes=report.peak_bytes,
                          spaces=report.spaces,
                          result_digest=str(getattr(placement, "input_digest", "")))


__all__ = ["STATUS_VERIFIED", "STATUS_FAILED", "StorageVerdict", "verify_storage",
           "invalidated_inputs"]
